use crate::hnsw::HnswIndex;
use crate::storage::VectorStorage;
use crate::types::{Neighbor, PointId};
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

    pub fn search(&self, query: &[f32], k: usize) -> Vec<Neighbor> {
        let guard = self.inner.read().expect("lock poisoned");
        guard.search(query, k)
    }

    pub fn len(&self) -> usize {
        let guard = self.inner.read().expect("lock poisoned");
        guard.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
}
