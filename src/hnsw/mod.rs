use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::path::Path;

use rand::{Rng, SeedableRng, rngs::StdRng, thread_rng};
use rayon::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::distance::{cosine_distance, l2};
use crate::storage::{InMemoryStorage, VectorStorage};
use crate::types::{IndexError, Metric, Neighbor, PointId};

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Node {
    pub layers: Vec<Vec<PointId>>, // index by layer
}

impl Node {
    pub fn new(max_layer: usize) -> Self {
        let layers = (0..=max_layer).map(|_| Vec::new()).collect();
        Self { layers }
    }
}

fn sample_level<R: Rng + ?Sized>(m: usize, rng: &mut R) -> usize {
    let mut level = 0;
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
    /// When set, makes level sampling reproducible for a fixed insertion order.
    #[serde(default)]
    pub level_seed: Option<u64>,
    /// Logically deleted points. Tombstoned nodes remain traversable until compaction.
    #[serde(default)]
    deleted: HashSet<PointId>,
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
        Self::try_new(m, m_max0, ef_construction, ef_search, metric, storage)
            .expect("invalid HNSW configuration")
    }

    /// Construct an index after validating parameters.
    pub fn try_new(
        m: usize,
        m_max0: usize,
        ef_construction: usize,
        ef_search: usize,
        metric: Metric,
        storage: S,
    ) -> Result<Self, IndexError> {
        if m == 0 {
            return Err(IndexError::InvalidConfiguration(
                "m must be greater than zero",
            ));
        }
        if m_max0 == 0 {
            return Err(IndexError::InvalidConfiguration(
                "m_max0 must be greater than zero",
            ));
        }
        if ef_construction == 0 {
            return Err(IndexError::InvalidConfiguration(
                "ef_construction must be greater than zero",
            ));
        }
        if ef_search == 0 {
            return Err(IndexError::InvalidConfiguration(
                "ef_search must be greater than zero",
            ));
        }

        Ok(Self {
            m,
            m_max0,
            ef_construction,
            ef_search,
            metric,
            entry_point: None,
            entry_point_level: 0,
            nodes: HashMap::new(),
            storage,
            level_seed: None,
            deleted: HashSet::new(),
        })
    }

    /// Use deterministic level sampling for reproducible construction.
    ///
    /// The seed must be configured before the first insertion.
    pub fn with_level_seed(mut self, seed: u64) -> Self {
        assert!(
            self.nodes.is_empty(),
            "level seed must be set before insertion"
        );
        self.level_seed = Some(seed);
        self
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
        self.try_insert(vector).expect("invalid vector")
    }

    /// Insert a vector after validating its dimension and values.
    pub fn try_insert(&mut self, vector: Vec<f32>) -> Result<PointId, IndexError> {
        self.validate_vector(&vector)?;
        let had_active_points = self.active_len() > 0;
        let id = self.storage.push(vector);
        let level = match self.level_seed {
            Some(seed) => {
                let mut rng =
                    StdRng::seed_from_u64(seed ^ (id as u64).wrapping_mul(0x9E3779B97F4A7C15));
                sample_level(self.m, &mut rng)
            }
            None => sample_level(self.m, &mut thread_rng()),
        };
        let node = Node::new(level);
        self.nodes.insert(id, node);
        if self.entry_point.is_none() || !had_active_points {
            self.entry_point = Some(id);
            self.entry_point_level = level;
            return Ok(id);
        }

        // Greedy descent on upper layers to find an entry for layers up to `level`.
        let mut entry = self.entry_point.unwrap();
        if level < self.entry_point_level {
            for l in (level + 1..=self.entry_point_level).rev() {
                let candidates =
                    self.search_layer_internal(entry, l, self.storage.get(id).unwrap().as_ref(), 1);
                if let Some(best) = candidates.first() {
                    entry = best.id;
                }
            }
        }

        // Insert connections layer by layer down to 0.
        for l in (0..=level).rev() {
            let ef = self.ef_construction;
            let candidates =
                self.search_layer_internal(entry, l, self.storage.get(id).unwrap().as_ref(), ef);
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
                if let Some(n_node) = self.nodes.get_mut(&n_id)
                    && l < n_node.layers.len()
                    && !n_node.layers[l].contains(&id)
                {
                    n_node.layers[l].push(id);
                }

                // Prune neighbor list after potential insertion.
                let current = self
                    .nodes
                    .get(&n_id)
                    .and_then(|n| n.layers.get(l))
                    .cloned()
                    .unwrap_or_default();
                let pruned = self.prune_neighbor_list(n_id, &current, limit);
                if let Some(n_node) = self.nodes.get_mut(&n_id)
                    && l < n_node.layers.len()
                {
                    n_node.layers[l] = pruned;
                }
            }
        }

        // If the new node reaches higher than current entry, it becomes the new entry point.
        if level > self.entry_point_level {
            self.entry_point = Some(id);
            self.entry_point_level = level;
        }
        Ok(id)
    }

    /// Logically delete a point while retaining its graph edges for traversal.
    pub fn delete(&mut self, id: PointId) -> Result<(), IndexError> {
        if !self.nodes.contains_key(&id) || self.storage.get(id).is_none() {
            return Err(IndexError::PointNotFound { id });
        }
        if !self.deleted.insert(id) {
            return Err(IndexError::PointAlreadyDeleted { id });
        }
        Ok(())
    }

    pub fn is_deleted(&self, id: PointId) -> bool {
        self.deleted.contains(&id)
    }

    pub fn total_len(&self) -> usize {
        self.storage.len()
    }

    pub fn active_len(&self) -> usize {
        self.storage.len().saturating_sub(self.deleted.len())
    }

    pub fn deleted_len(&self) -> usize {
        self.deleted.len()
    }

    /// Rebuild active points into an empty storage backend.
    ///
    /// Compaction physically removes tombstones and returns an old-to-new point ID map.
    pub fn compact_into<T: VectorStorage>(
        &self,
        storage: T,
    ) -> Result<(HnswIndex<T>, HashMap<PointId, PointId>), IndexError> {
        if !storage.is_empty() {
            return Err(IndexError::InvalidConfiguration(
                "compaction destination must be empty",
            ));
        }

        let mut compacted = HnswIndex::try_new(
            self.m,
            self.m_max0,
            self.ef_construction,
            self.ef_search,
            self.metric,
            storage,
        )?;
        compacted.level_seed = self.level_seed;

        let mut id_map = HashMap::with_capacity(self.active_len());
        for old_id in 0..self.storage.len() {
            if self.deleted.contains(&old_id) {
                continue;
            }
            let vector = self
                .storage
                .get_owned(old_id)
                .ok_or(IndexError::PointNotFound { id: old_id })?;
            let new_id = compacted.try_insert(vector)?;
            id_map.insert(old_id, new_id);
        }
        Ok((compacted, id_map))
    }

    /// Search for k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<Neighbor> {
        self.try_search(query, k).expect("invalid query vector")
    }

    /// Search after validating the query dimension and values.
    pub fn try_search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor>, IndexError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        if !self.storage.is_empty() {
            self.validate_vector(query)?;
        }
        let Some(mut entry) = self.entry_point else {
            return Ok(Vec::new());
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
        Ok(results
            .into_iter()
            .map(|sp| Neighbor {
                id: sp.id,
                distance: sp.distance,
            })
            .collect())
    }

    pub fn validate_vector(&self, vector: &[f32]) -> Result<(), IndexError> {
        if vector.is_empty() {
            return Err(IndexError::EmptyVector);
        }
        if let Some(expected) = self.storage.dim()
            && vector.len() != expected
        {
            return Err(IndexError::DimensionMismatch {
                expected,
                actual: vector.len(),
            });
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(IndexError::NonFiniteValue);
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
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
        let mut idx =
            HnswIndex::new(8, 16, 32, 32, Metric::L2, InMemoryStorage::new()).with_level_seed(99);
        idx.insert(vec![0.0, 0.0]);
        idx.insert(vec![1.0, 0.0]);
        idx.insert(vec![0.0, 1.0]);
        idx.delete(1).unwrap();

        let path = env::temp_dir().join("vector-db-hnsw.json");
        idx.save_to_json(&path).expect("save");
        let loaded: HnswIndex<InMemoryStorage> = HnswIndex::load_from_json(&path).expect("load");
        let _ = fs::remove_file(&path);

        assert_eq!(loaded.level_seed, Some(99));
        assert!(loaded.is_deleted(1));
        let res = loaded.search(&[0.9, 0.1], 2);
        assert!(!res.is_empty());
        assert!(res.iter().all(|neighbor| neighbor.id != 1));
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

    #[test]
    fn seeded_construction_is_reproducible() {
        let data: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32, (i % 7) as f32]).collect();
        let build = || {
            let mut index = HnswIndex::new(4, 8, 16, 16, Metric::L2, InMemoryStorage::new())
                .with_level_seed(42);
            for vector in data.iter().cloned() {
                index.insert(vector);
            }
            index
        };

        let first = build();
        let second = build();
        assert_eq!(first.entry_point, second.entry_point);
        assert_eq!(first.entry_point_level, second.entry_point_level);
        assert_eq!(first.nodes, second.nodes);
    }

    #[test]
    fn fallible_api_rejects_invalid_vectors() {
        let mut index = HnswIndex::new(8, 16, 32, 32, Metric::L2, InMemoryStorage::new());
        assert_eq!(index.try_insert(Vec::new()), Err(IndexError::EmptyVector));
        index.try_insert(vec![0.0, 1.0]).unwrap();
        assert_eq!(
            index.try_insert(vec![0.0]),
            Err(IndexError::DimensionMismatch {
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(
            index.try_search(&[f32::NAN, 0.0], 1),
            Err(IndexError::NonFiniteValue)
        );
    }

    #[test]
    fn deleted_points_are_traversed_but_not_returned() {
        let mut index =
            HnswIndex::new(8, 16, 32, 32, Metric::L2, InMemoryStorage::new()).with_level_seed(7);
        for value in 0..5 {
            index.insert(vec![value as f32, 0.0]);
        }

        index.delete(2).unwrap();
        let results = index.search(&[2.0, 0.0], 5);
        assert_eq!(index.total_len(), 5);
        assert_eq!(index.active_len(), 4);
        assert!(results.iter().all(|neighbor| neighbor.id != 2));
        assert_eq!(results.len(), 4);
        assert_eq!(
            index.delete(2),
            Err(IndexError::PointAlreadyDeleted { id: 2 })
        );
        assert_eq!(index.delete(99), Err(IndexError::PointNotFound { id: 99 }));
    }

    #[test]
    fn compaction_removes_tombstones_and_returns_id_map() {
        let mut index =
            HnswIndex::new(8, 16, 32, 32, Metric::L2, InMemoryStorage::new()).with_level_seed(11);
        for value in 0..4 {
            index.insert(vec![value as f32, 0.0]);
        }
        index.delete(1).unwrap();

        let (compacted, id_map) = index.compact_into(InMemoryStorage::new()).unwrap();
        assert_eq!(compacted.total_len(), 3);
        assert_eq!(compacted.deleted_len(), 0);
        assert_eq!(id_map.get(&0), Some(&0));
        assert_eq!(id_map.get(&2), Some(&1));
        assert_eq!(id_map.get(&3), Some(&2));
        assert!(!id_map.contains_key(&1));
        assert_eq!(compacted.search(&[2.0, 0.0], 1)[0].id, 1);
    }

    #[test]
    fn insertion_after_deleting_every_point_establishes_a_new_entry() {
        let mut index =
            HnswIndex::new(8, 16, 32, 32, Metric::L2, InMemoryStorage::new()).with_level_seed(13);
        let old_id = index.insert(vec![0.0, 0.0]);
        index.delete(old_id).unwrap();

        let new_id = index.insert(vec![1.0, 1.0]);
        assert_eq!(index.entry_point, Some(new_id));
        assert_eq!(index.search(&[1.0, 1.0], 1)[0].id, new_id);
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
            if !self.deleted.contains(&entry_id) {
                result.push(ep);
            }
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
                        if !self.deleted.contains(&n_id) {
                            result.push(sp);
                            if result.len() > ef {
                                result.pop(); // drop the farthest
                            }
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
        uniq.retain(|id| !self.deleted.contains(id));
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
        serde_json::to_writer(writer, self).map_err(|e| io::Error::other(format!("serialize: {e}")))
    }

    /// Load an index previously saved with `save_to_json`.
    pub fn load_from_json<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(|e| io::Error::other(format!("deserialize: {e}")))
    }
}

impl Default for HnswIndex<InMemoryStorage> {
    fn default() -> Self {
        Self::new(16, 32, 64, 32, Metric::L2, InMemoryStorage::new())
    }
}
