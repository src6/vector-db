use crate::hnsw::HnswIndex;
use crate::storage::VectorStorage;
use crate::types::{Metric, Neighbor, PointId};

/// Two-segment index: immutable segment for reads, mutable for new writes.
pub struct SegmentedIndex<SI: VectorStorage, SM: VectorStorage + Default> {
    immutable: HnswIndex<SI>,
    mutable: HnswIndex<SM>,
    params: (usize, usize, usize, usize, Metric),
}

impl<SI, SM> SegmentedIndex<SI, SM>
where
    SI: VectorStorage,
    SM: VectorStorage + Default,
{
    pub fn new(
        immutable: HnswIndex<SI>,
        m: usize,
        m_max0: usize,
        ef_construction: usize,
        ef_search: usize,
        metric: Metric,
    ) -> Self {
        let mutable = HnswIndex::new(m, m_max0, ef_construction, ef_search, metric, SM::default());
        Self {
            immutable,
            mutable,
            params: (m, m_max0, ef_construction, ef_search, metric),
        }
    }

    pub fn insert(&mut self, vector: Vec<f32>) -> PointId {
        let base = self.immutable.storage.len();
        let id = self.mutable.insert(vector);
        base + id
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<Neighbor> {
        let base = self.immutable.storage.len();
        let mut candidates = self.immutable.search(query, k);
        let mut fresh = self.mutable.search(query, k);
        for n in fresh.iter_mut() {
            n.id += base;
        }
        candidates.extend(fresh);
        candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        candidates.truncate(k);
        candidates
    }

    /// Merge mutable segment into immutable and reset mutable.
    pub fn flush_mutable_into_immutable(&mut self) {
        let len = self.mutable.storage.len();
        for i in 0..len {
            if let Some(vec) = self.mutable.storage.get_owned(i) {
                self.immutable.insert(vec);
            }
        }
        let (m, m_max0, ef_c, ef_s, metric) = self.params;
        self.mutable = HnswIndex::new(m, m_max0, ef_c, ef_s, metric, SM::default());
    }

    pub fn immutable_len(&self) -> usize {
        self.immutable.storage.len()
    }

    pub fn mutable_len(&self) -> usize {
        self.mutable.storage.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::InMemoryStorage;

    #[test]
    fn search_merges_segments() {
        let base_idx = HnswIndex::new(8, 16, 16, 32, Metric::L2, InMemoryStorage::new());
        let mut seg: SegmentedIndex<InMemoryStorage, InMemoryStorage> =
            SegmentedIndex::new(base_idx, 8, 16, 16, 32, Metric::L2);
        seg.insert(vec![0.0, 0.0]);
        // add into immutable to simulate persisted data
        seg.flush_mutable_into_immutable();
        seg.insert(vec![1.0, 0.0]); // stays mutable

        let res = seg.search(&[0.1, 0.0], 2);
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].id, 0);
    }

    #[test]
    fn flush_moves_mutable() {
        let base_idx = HnswIndex::new(8, 16, 16, 32, Metric::L2, InMemoryStorage::new());
        let mut seg: SegmentedIndex<InMemoryStorage, InMemoryStorage> =
            SegmentedIndex::new(base_idx, 8, 16, 16, 32, Metric::L2);
        seg.insert(vec![1.0, 0.0]);
        assert_eq!(seg.mutable_len(), 1);
        seg.flush_mutable_into_immutable();
        assert_eq!(seg.mutable_len(), 0);
        assert_eq!(seg.immutable_len(), 1);
    }
}
