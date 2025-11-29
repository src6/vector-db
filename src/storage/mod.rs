mod in_memory;

pub use in_memory::InMemoryStorage;

use crate::types::PointId;

pub trait VectorStorage {
    fn push(&mut self, vector: Vec<f32>) -> PointId;
    fn get(&self, id: PointId) -> Option<&[f32]>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
