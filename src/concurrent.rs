use crate::hnsw::HnswIndex;
use crate::storage::{InMemoryStorage, VectorStorage};
use crate::types::{IndexError, Neighbor, PointId};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Thread-safe wrapper around `HnswIndex` using coarse-grained locking.
#[derive(Clone)]
pub struct ConcurrentIndex<S: VectorStorage> {
    inner: Arc<RwLock<HnswIndex<S>>>,
}

impl<S> ConcurrentIndex<S>
where
    S: VectorStorage + Send + Sync,
{
    pub fn new(index: HnswIndex<S>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(index)),
        }
    }

    pub fn insert(&self, vector: Vec<f32>) -> PointId {
        let mut guard = self.inner.write().expect("lock poisoned");
        guard.insert(vector)
    }

    pub fn try_insert(&self, vector: Vec<f32>) -> Result<PointId, IndexError> {
        let mut guard = self.inner.write().expect("lock poisoned");
        guard.try_insert(vector)
    }

    /// Validate a whole batch before inserting it under one write lock.
    pub fn try_insert_batch(&self, vectors: Vec<Vec<f32>>) -> Result<Vec<PointId>, IndexError> {
        let mut guard = self.inner.write().expect("lock poisoned");
        let expected_dimension = guard
            .storage
            .dim()
            .or_else(|| vectors.first().map(Vec::len));
        for vector in &vectors {
            guard.validate_vector(vector)?;
            if let Some(expected) = expected_dimension
                && vector.len() != expected
            {
                return Err(IndexError::DimensionMismatch {
                    expected,
                    actual: vector.len(),
                });
            }
        }
        vectors
            .into_iter()
            .map(|vector| guard.try_insert(vector))
            .collect()
    }

    /// Insert a batch of vectors in parallel, leveraging coarse lock per insert.
    pub fn insert_batch_parallel<I>(&self, vectors: I) -> Vec<PointId>
    where
        I: IntoParallelIterator<Item = Vec<f32>>,
    {
        vectors.into_par_iter().map(|v| self.insert(v)).collect()
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<Neighbor> {
        let guard = self.inner.read().expect("lock poisoned");
        guard.search(query, k)
    }

    pub fn try_search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor>, IndexError> {
        let guard = self.inner.read().expect("lock poisoned");
        guard.try_search(query, k)
    }

    pub fn try_search_batch_parallel(
        &self,
        queries: Vec<Vec<f32>>,
        k: usize,
    ) -> Result<Vec<Vec<Neighbor>>, IndexError> {
        queries
            .into_par_iter()
            .map(|query| self.try_search(&query, k))
            .collect()
    }

    pub fn delete(&self, id: PointId) -> Result<(), IndexError> {
        let mut guard = self.inner.write().expect("lock poisoned");
        guard.delete(id)
    }

    pub fn search_batch_parallel<I>(&self, queries: I, k: usize) -> Vec<Vec<Neighbor>>
    where
        I: IntoParallelIterator<Item = Vec<f32>>,
    {
        queries
            .into_par_iter()
            .map(|q| self.search(&q, k))
            .collect()
    }

    pub fn len(&self) -> usize {
        let guard = self.inner.read().expect("lock poisoned");
        guard.storage.len()
    }

    pub fn active_len(&self) -> usize {
        let guard = self.inner.read().expect("lock poisoned");
        guard.active_len()
    }

    pub fn deleted_len(&self) -> usize {
        let guard = self.inner.read().expect("lock poisoned");
        guard.deleted_len()
    }

    pub fn counts(&self) -> (usize, usize, usize) {
        let guard = self.inner.read().expect("lock poisoned");
        (guard.total_len(), guard.active_len(), guard.deleted_len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl ConcurrentIndex<InMemoryStorage> {
    /// Rebuild the index without tombstones while holding the write lock.
    pub fn compact(&self) -> Result<(usize, HashMap<PointId, PointId>), IndexError> {
        let mut guard = self.inner.write().expect("lock poisoned");
        let removed = guard.deleted_len();
        let (compacted, id_map) = guard.compact_into(InMemoryStorage::new())?;
        *guard = compacted;
        Ok((removed, id_map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::InMemoryStorage;
    use crate::types::Metric;
    use std::thread;

    #[test]
    fn concurrent_inserts_are_visible() {
        let index = HnswIndex::new(8, 16, 16, 16, Metric::L2, InMemoryStorage::new());
        let shared = ConcurrentIndex::new(index);

        let mut handles = Vec::new();
        for t in 0..4 {
            let idx = shared.clone();
            handles.push(thread::spawn(move || {
                for i in 0..25 {
                    idx.insert(vec![t as f32, i as f32]);
                }
            }));
        }
        for h in handles {
            h.join().expect("thread failed");
        }

        assert_eq!(shared.len(), 100);
        let res = shared.search(&[0.0, 0.0], 5);
        assert!(!res.is_empty());
    }

    #[test]
    fn batch_parallel_search_and_insert_work() {
        let index = HnswIndex::new(8, 16, 16, 16, Metric::L2, InMemoryStorage::new());
        let shared = ConcurrentIndex::new(index);
        let data: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32, 0.0]).collect();
        let _ids = shared.insert_batch_parallel(data.clone());
        assert_eq!(shared.len(), 50);
        let queries: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32 + 0.1, 0.0]).collect();
        let res = shared.search_batch_parallel(queries, 1);
        assert_eq!(res.len(), 10);
    }

    #[test]
    fn invalid_batch_is_rejected_before_any_insert() {
        let index = HnswIndex::new(8, 16, 16, 16, Metric::L2, InMemoryStorage::new());
        let shared = ConcurrentIndex::new(index);
        let result = shared.try_insert_batch(vec![vec![1.0, 2.0], vec![3.0]]);
        assert_eq!(
            result,
            Err(IndexError::DimensionMismatch {
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(shared.len(), 0);
    }
}
