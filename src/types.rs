/// Unique identifier for a stored vector.
pub type PointId = usize;

/// Supported distance metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    L2,
    Cosine,
}

/// Neighbor with distance, useful for search results.
#[derive(Debug, Clone, PartialEq)]
pub struct Neighbor {
    pub id: PointId,
    pub distance: f32,
}
