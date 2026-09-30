/// Unique identifier for a stored vector.
pub type PointId = usize;

/// Supported distance metrics.
use serde::{Deserialize, Serialize};
use std::fmt;

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

/// Errors returned by the fallible index API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexError {
    InvalidConfiguration(&'static str),
    EmptyVector,
    DimensionMismatch { expected: usize, actual: usize },
    NonFiniteValue,
}

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => write!(f, "invalid configuration: {message}"),
            Self::EmptyVector => write!(f, "vectors must contain at least one dimension"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::NonFiniteValue => write!(f, "vectors must contain only finite values"),
        }
    }
}

impl std::error::Error for IndexError {}
