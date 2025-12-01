mod in_memory;
mod quantized;

pub use in_memory::InMemoryStorage;
pub use quantized::{QuantizedStorage, ScalarQuantizerConfig};

use crate::types::PointId;
use std::borrow::Cow;

pub trait VectorStorage {
    fn push(&mut self, vector: Vec<f32>) -> PointId;
    fn get(&self, id: PointId) -> Option<Cow<'_, [f32]>>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
