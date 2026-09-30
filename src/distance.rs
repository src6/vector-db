pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0; // treat zero vector as max distance
    }
    let denom = na.sqrt() * nb.sqrt();
    1.0 - (dot / denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_returns_squared_euclidean_distance() {
        assert_eq!(l2(&[1.0, 2.0], &[4.0, 6.0]), 25.0);
    }

    #[test]
    fn cosine_handles_parallel_orthogonal_and_zero_vectors() {
        assert!((cosine_distance(&[1.0, 0.0], &[2.0, 0.0])).abs() < f32::EPSILON);
        assert_eq!(cosine_distance(&[1.0, 0.0], &[0.0, 1.0]), 1.0);
        assert_eq!(cosine_distance(&[0.0, 0.0], &[1.0, 0.0]), 1.0);
    }
}
