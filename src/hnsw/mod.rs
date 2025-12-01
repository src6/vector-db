use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use rand::{Rng, thread_rng};
use rayon::prelude::*;

use crate::distance::{cosine_distance, l2};
use crate::storage::{InMemoryStorage, VectorStorage};
use crate::types::{Metric, Neighbor, PointId};

#[derive(Debug, Clone)]
struct ScoredPoint {
    id: PointId,
    distance: f32,
}

impl PartialEq for ScoredPoint {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.distance.to_bits() == other.distance.to_bits()
    }
}

impl Eq for ScoredPoint {}

impl PartialOrd for ScoredPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredPoint {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.distance.partial_cmp(&other.distance) {
            Some(ord) => ord.then_with(|| self.id.cmp(&other.id)),
            None => Ordering::Equal,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub layers: Vec<Vec<PointId>>, // index by layer
}

impl Node {
    pub fn new(max_layer: usize) -> Self {
        let layers = (0..=max_layer).map(|_| Vec::new()).collect();
        Self { layers }
    }
}

fn sample_level(m: usize) -> usize {
    if m == 0 {
        return 0;
    }
    let mut level = 0;
    let mut rng = thread_rng();
    let p = 1.0f32 / (m as f32); // decay factor; higher m -> shorter tail
    while rng.gen_range(0.0..1.0) < p {
        level += 1;
    }
    level
}

fn sort_by_distance(mut items: Vec<ScoredPoint>) -> Vec<ScoredPoint> {
    items.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    items
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "S: Serialize + DeserializeOwned")]
pub struct HnswIndex<S: VectorStorage = InMemoryStorage> {
    pub m: usize,
    pub m_max0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub metric: Metric,
    pub entry_point: Option<PointId>,
    pub entry_point_level: usize,
    pub nodes: HashMap<PointId, Node>,
    pub storage: S,
}

impl<S: VectorStorage> HnswIndex<S> {
    pub fn new(
        m: usize,
        m_max0: usize,
        ef_construction: usize,
        ef_search: usize,
        metric: Metric,
        storage: S,
    ) -> Self {
        Self {
            m,
            m_max0,
            ef_construction,
            ef_search,
            metric,
            entry_point: None,
            entry_point_level: 0,
            nodes: HashMap::new(),
            storage,
        }
    }

    /// Parallel search for many queries using rayon.
    pub fn search_batch_parallel<I>(&self, queries: I, k: usize) -> Vec<Vec<Neighbor>>
    where
        I: rayon::iter::IntoParallelIterator<Item = Vec<f32>>,
        S: Sync,
    {
        queries
            .into_par_iter()
            .map(|q| self.search(&q, k))
            .collect()
    }

    /// Insert a vector; returns its id.
    pub fn insert(&mut self, vector: Vec<f32>) -> PointId {
        let id = self.storage.push(vector);
        let level = sample_level(self.m);
        let node = Node::new(level);
        self.nodes.insert(id, node);
        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            self.entry_point_level = level;
            return id;
        }

        // Greedy descent on upper layers to find an entry for layers up to `level`.
        let mut entry = self.entry_point.unwrap();
        if level < self.entry_point_level {
            for l in (level + 1..=self.entry_point_level).rev() {
                let candidates = self.search_layer_internal(
                    entry,
                    l,
                    self.storage.get(id).unwrap().as_ref(),
                    1,
                );
                if let Some(best) = candidates.first() {
                    entry = best.id;
                }
            }
        }

        // Insert connections layer by layer down to 0.
        for l in (0..=level).rev() {
            let ef = self.ef_construction;
            let candidates = self.search_layer_internal(
                entry,
                l,
                self.storage.get(id).unwrap().as_ref(),
                ef,
            );
            let max_m = if l == 0 { self.m_max0 } else { self.m };
            let neighbors = self.select_neighbors(candidates, max_m);

            // Connect new node to selected neighbors on this layer.
            if let Some(node) = self.nodes.get_mut(&id) {
                node.layers[l] = neighbors.clone();
            }

            // For each neighbor, add reverse edge and prune if needed.
            for &n_id in &neighbors {
                let limit = if l == 0 { self.m_max0 } else { self.m };

                // Add reverse edge.
                if let Some(n_node) = self.nodes.get_mut(&n_id) {
                    if l < n_node.layers.len() && !n_node.layers[l].contains(&id) {
                        n_node.layers[l].push(id);
                    }
                }

                // Prune neighbor list after potential insertion.
                let current = self
                    .nodes
                    .get(&n_id)
                    .and_then(|n| n.layers.get(l))
                    .cloned()
                    .unwrap_or_default();
                let pruned = self.prune_neighbor_list(n_id, &current, limit);
                if let Some(n_node) = self.nodes.get_mut(&n_id) {
                    if l < n_node.layers.len() {
                        n_node.layers[l] = pruned;
                    }
                }
            }
        }

        // If the new node reaches higher than current entry, it becomes the new entry point.
        if level > self.entry_point_level {
            self.entry_point = Some(id);
            self.entry_point_level = level;
        }
        id
    }

    /// Search for k nearest neighbors (squared L2 for now).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<Neighbor> {
        let Some(mut entry) = self.entry_point else {
            return Vec::new();
        };

        // Greedy descent on upper layers (ef = 1) to pick an entry to layer 0.
        for l in (1..=self.entry_point_level).rev() {
            let candidates = self.search_layer_internal(entry, l, query, 1);
            if let Some(best) = candidates.first() {
                entry = best.id;
            }
        }

        // Layer 0 beam search with ef_search.
        let mut results = self.search_layer_internal(entry, 0, query, self.ef_search);
        results = sort_by_distance(results);
        results.truncate(k);
        results
            .into_iter()
            .map(|sp| Neighbor {
                id: sp.id,
                distance: sp.distance,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{InMemoryStorage, QuantizedStorage, ScalarQuantizerConfig};
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use std::env;
    use std::fs;

    fn brute_force(query: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<PointId> {
        let mut scored: Vec<(PointId, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, l2(query, v)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.truncate(k);
        scored.into_iter().map(|(i, _)| i).collect()
    }

    #[test]
    fn search_matches_brute_force_small_set() {
        let mut idx = HnswIndex::new(8, 16, 32, 32, Metric::L2, InMemoryStorage::new());
        let data = vec![
            vec![0.0f32, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![2.0, 0.0],
        ];

        for v in data.iter().cloned() {
            idx.insert(v);
        }

        let query = [0.9f32, 0.1];
        let k = 3;
        let brute = brute_force(&query, &data, k);
        let hnsw = idx.search(&query, k);
        let h_ids: Vec<_> = hnsw.into_iter().map(|n| n.id).collect();

        // Require that all brute-force top-k are present in HNSW results for this tiny set.
        for id in brute {
            assert!(h_ids.contains(&id));
        }
    }

    #[test]
    fn search_empty_returns_empty() {
        let idx = HnswIndex::new(8, 16, 32, 32, Metric::L2, InMemoryStorage::new());
        let res = idx.search(&[0.0f32, 0.0], 3);
        assert!(res.is_empty());
    }

    #[test]
    fn search_single_point() {
        let mut idx = HnswIndex::new(8, 16, 32, 32, Metric::L2, InMemoryStorage::new());
        idx.insert(vec![5.0, -1.0]);
        let res = idx.search(&[5.0, -1.0], 1);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, 0);
    }

    #[test]
    fn random_small_dataset_matches_bruteforce() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut idx = HnswIndex::new(8, 16, 64, 64, Metric::L2, InMemoryStorage::new());
        let dim = 8;
        let n = 50;
        let mut data = Vec::new();
        for _ in 0..n {
            let vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            data.push(vec.clone());
            idx.insert(vec);
        }

        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let k = 5;
        let brute = brute_force(&query, &data, k);
        let hnsw = idx.search(&query, k);
        let h_ids: Vec<_> = hnsw.into_iter().map(|n| n.id).collect();

        for id in brute {
            assert!(h_ids.contains(&id));
        }
    }

    #[test]
    fn persistence_round_trip() {
        let mut idx = HnswIndex::new(8, 16, 32, 32, Metric::L2, InMemoryStorage::new());
        idx.insert(vec![0.0, 0.0]);
        idx.insert(vec![1.0, 0.0]);
        idx.insert(vec![0.0, 1.0]);

        let path = env::temp_dir().join("vector-db-hnsw.json");
        idx.save_to_json(&path).expect("save");
        let loaded: HnswIndex<InMemoryStorage> =
            HnswIndex::load_from_json(&path).expect("load");
        let _ = fs::remove_file(&path);

        let res = loaded.search(&[0.9, 0.1], 2);
        assert!(!res.is_empty());
    }

    #[test]
    fn quantized_storage_searches() {
        let sample = vec![vec![0.0f32, 0.0], vec![2.0, 2.0]];
        let cfg = ScalarQuantizerConfig::from_sample(&sample).unwrap();
        let store = QuantizedStorage::new(cfg);
        let mut idx = HnswIndex::new(8, 16, 32, 32, Metric::L2, store);
        let points = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        for p in points {
            idx.insert(p);
        }
        let res = idx.search(&[1.1, 1.0], 2);
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].id, 1);
    }

    #[test]
    fn batch_parallel_searches_return_results() {
        let mut idx = HnswIndex::new(8, 16, 16, 32, Metric::L2, InMemoryStorage::new());
        for i in 0..20 {
            idx.insert(vec![i as f32, 0.0]);
        }
        let queries: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32 + 0.3, 0.0]).collect();
        let results = idx.search_batch_parallel(queries, 1);
        assert_eq!(results.len(), 5);
        for (i, res) in results.iter().enumerate() {
            assert_eq!(res[0].id, i);
        }
    }
}

impl<S: VectorStorage> HnswIndex<S> {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            Metric::L2 => l2(a, b),
            Metric::Cosine => cosine_distance(a, b),
        }
    }

    /// Search a given layer starting from entry point, keeping up to `ef` closest candidates.
    fn search_layer_internal(
        &self,
        entry_id: PointId,
        layer: usize,
        query: &[f32],
        ef: usize,
    ) -> Vec<ScoredPoint> {
        let mut visited = HashSet::new();
        let mut candidate = BinaryHeap::new(); // min-heap via Reverse
        let mut result = BinaryHeap::new(); // max-heap, keeps worst on top

        if let Some(vec) = self.storage.get(entry_id) {
            let dist = self.distance(query, vec.as_ref());
            let ep = ScoredPoint {
                id: entry_id,
                distance: dist,
            };
            visited.insert(entry_id);
            candidate.push(Reverse(ep.clone()));
            result.push(ep);
        }

        while let Some(Reverse(curr)) = candidate.pop() {
            let worst = result.peek().map(|p| p.distance).unwrap_or(f32::MAX);
            if result.len() >= ef && curr.distance > worst {
                break;
            }

            if let Some(neighbors) = self.nodes.get(&curr.id).and_then(|n| n.layers.get(layer)) {
                for &n_id in neighbors {
                    if !visited.insert(n_id) {
                        continue;
                    }
                    if let Some(vec) = self.storage.get(n_id) {
                        let d = self.distance(query, vec.as_ref());
                        let sp = ScoredPoint {
                            id: n_id,
                            distance: d,
                        };
                        candidate.push(Reverse(sp.clone()));
                        result.push(sp);
                        if result.len() > ef {
                            result.pop(); // drop the farthest
                        }
                    }
                }
            }
        }

        result.into_iter().collect()
    }

    /// Heuristic neighbor selection favoring diversity.
    fn select_neighbors(&self, candidates: Vec<ScoredPoint>, max_m: usize) -> Vec<PointId> {
        let mut selected: Vec<PointId> = Vec::new();
        if candidates.is_empty() || max_m == 0 {
            return selected;
        }

        let by_dist = sort_by_distance(candidates);
        let mut remaining = Vec::new();

        for cand in by_dist.into_iter() {
            if selected.len() >= max_m {
                break;
            }
            let cand_vec = match self.storage.get(cand.id) {
                Some(v) => v,
                None => continue,
            };
            let mut good = true;
            for &sid in &selected {
                if let Some(sel_vec) = self.storage.get(sid) {
                    let dist = self.distance(cand_vec.as_ref(), sel_vec.as_ref());
                    if dist < cand.distance {
                        good = false;
                        break;
                    }
                }
            }
            if good {
                selected.push(cand.id);
            } else {
                remaining.push(cand);
            }
        }

        // Fallback: fill remaining slots with closest left-over candidates.
        if selected.len() < max_m {
            remaining.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
            for cand in remaining {
                if selected.len() >= max_m {
                    break;
                }
                if !selected.contains(&cand.id) {
                    selected.push(cand.id);
                }
            }
        }

        selected
    }

    /// Prune neighbor list to max_m closest to `src`.
    fn prune_neighbor_list(
        &self,
        src: PointId,
        neighbors: &[PointId],
        max_m: usize,
    ) -> Vec<PointId> {
        if max_m == 0 {
            return Vec::new();
        }
        let Some(src_vec) = self.storage.get(src) else {
            return neighbors.to_vec();
        };

        let mut uniq = neighbors.to_vec();
        uniq.sort_unstable();
        uniq.dedup();

        let mut scored: Vec<ScoredPoint> = uniq
            .into_iter()
            .filter_map(|nid| {
                self.storage.get(nid).map(|v| ScoredPoint {
                    id: nid,
                    distance: self.distance(src_vec.as_ref(), v.as_ref()),
                })
            })
            .collect();

        scored.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        scored.truncate(max_m);
        scored.into_iter().map(|s| s.id).collect()
    }
}

impl<S> HnswIndex<S>
where
    S: VectorStorage + Serialize + DeserializeOwned,
{
    /// Save the index (graph + storage) to a JSON file.
    pub fn save_to_json<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("serialize: {e}")))
    }

    /// Load an index previously saved with `save_to_json`.
    pub fn load_from_json<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("deserialize: {e}")))
    }
}

impl Default for HnswIndex<InMemoryStorage> {
    fn default() -> Self {
        Self::new(16, 32, 64, 32, Metric::L2, InMemoryStorage::new())
    }
}
