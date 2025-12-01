use super::VectorStorage;
use crate::types::PointId;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InMemoryStorage {
    data: Vec<Vec<f32>>,
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorStorage for InMemoryStorage {
    fn push(&mut self, vector: Vec<f32>) -> PointId {
        let id = self.data.len();
        self.data.push(vector);
        id
    }

    fn get(&self, id: PointId) -> Option<Cow<'_, [f32]>> {
        self.data.get(id).map(|v| Cow::Borrowed(v.as_slice()))
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}
