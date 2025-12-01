use super::VectorStorage;
use crate::types::PointId;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Simple per-dimension scalar quantizer using min/max scaling to i8.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarQuantizerConfig {
    mins: Vec<f32>,
    maxs: Vec<f32>,
}

impl ScalarQuantizerConfig {
    pub fn new(mins: Vec<f32>, maxs: Vec<f32>) -> Self {
        assert_eq!(
            mins.len(),
            maxs.len(),
            "quantizer mins/maxs dimension mismatch"
        );
        Self { mins, maxs }
    }

    /// Build a quantizer from a sample dataset by capturing per-dimension min/max.
    pub fn from_sample(sample: &[Vec<f32>]) -> Option<Self> {
        let first = sample.first()?;
        let dim = first.len();
        let mut mins = first.clone();
        let mut maxs = first.clone();
        for vec in sample.iter().skip(1) {
            assert_eq!(vec.len(), dim, "sample dimension mismatch");
            for (i, v) in vec.iter().enumerate() {
                if *v < mins[i] {
                    mins[i] = *v;
                }
                if *v > maxs[i] {
                    maxs[i] = *v;
                }
            }
        }
        Some(Self { mins, maxs })
    }

    pub fn dim(&self) -> usize {
        self.mins.len()
    }

    fn clamp_range(&self, dim: usize) -> f32 {
        (self.maxs[dim] - self.mins[dim]).max(f32::EPSILON)
    }

    fn quantize_value(&self, dim: usize, value: f32) -> i8 {
        let range = self.clamp_range(dim);
        let scaled = (value - self.mins[dim]) / range; // 0..1
        let centered = (scaled * 254.0) - 127.0; // map into [-127,127]
        centered.round().clamp(-127.0, 127.0) as i8
    }

    fn dequantize_value(&self, dim: usize, code: i8) -> f32 {
        let range = self.clamp_range(dim);
        let scaled = (code as f32 + 127.0) / 254.0; // back to 0..1
        (scaled * range) + self.mins[dim]
    }
}

/// Quantized storage that keeps i8 codes and dequantizes on read.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedStorage {
    codes: Vec<Vec<i8>>,
    config: ScalarQuantizerConfig,
}

impl QuantizedStorage {
    pub fn new(config: ScalarQuantizerConfig) -> Self {
        Self {
            codes: Vec::new(),
            config,
        }
    }

    fn encode(&self, vector: &[f32]) -> Vec<i8> {
        assert_eq!(
            vector.len(),
            self.config.dim(),
            "vector dimension does not match quantizer"
        );
        vector
            .iter()
            .enumerate()
            .map(|(i, v)| self.config.quantize_value(i, *v))
            .collect()
    }

    fn decode(&self, codes: &[i8]) -> Vec<f32> {
        codes
            .iter()
            .enumerate()
            .map(|(i, c)| self.config.dequantize_value(i, *c))
            .collect()
    }
}

impl VectorStorage for QuantizedStorage {
    fn push(&mut self, vector: Vec<f32>) -> PointId {
        let id = self.codes.len();
        let encoded = self.encode(&vector);
        self.codes.push(encoded);
        id
    }

    fn get(&self, id: PointId) -> Option<Cow<'_, [f32]>> {
        let codes = self.codes.get(id)?;
        let decoded = self.decode(codes);
        Some(Cow::Owned(decoded))
    }

    fn len(&self) -> usize {
        self.codes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_round_trip() {
        let sample = vec![vec![0.0f32, 10.0], vec![2.0, 20.0]];
        let cfg = ScalarQuantizerConfig::from_sample(&sample).unwrap();
        let mut store = QuantizedStorage::new(cfg);

        let id = store.push(vec![1.0, 15.0]);
        let decoded = store.get(id).unwrap();
        assert_eq!(decoded.as_ref().len(), 2);
        assert!((decoded[0] - 1.0).abs() < 0.2);
        assert!((decoded[1] - 15.0).abs() < 0.2);
    }
}
