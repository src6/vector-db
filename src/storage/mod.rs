mod in_memory;
mod quantized;
mod mmap;

pub use in_memory::InMemoryStorage;
pub use quantized::{QuantizedStorage, ScalarQuantizerConfig};
pub use mmap::MmapStorage;

use crate::types::PointId;
use std::borrow::Cow;

pub trait VectorStorage {
    fn push(&mut self, vector: Vec<f32>) -> PointId;
    fn get(&self, id: PointId) -> Option<Cow<'_, [f32]>>;
    fn len(&self) -> usize;
    fn dim(&self) -> Option<usize> {
        None
    }
    fn get_owned(&self, id: PointId) -> Option<Vec<f32>> {
        self.get(id).map(|c| c.into_owned())
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
