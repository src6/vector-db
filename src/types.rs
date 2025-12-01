/// Unique identifier for a stored vector.
pub type PointId = usize;

/// Supported distance metrics.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    L2,
    Cosine,
}

/// Neighbor with distance, useful for search results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Neighbor {
    pub id: PointId,
    pub distance: f32,
}
