use half::bf16;

/// Enum to represent different distance metrics.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DistanceMetric {
    Euclidean,
    DotProduct,
    Hamming,
}

pub fn dot_product_f32(v1: &[f32], v2: &[f32]) -> f32 {
    if v1.len() != v2.len() {
        panic!("Vectors must be of the same length");
    }
    let dp: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    dp * -1.0
}

pub fn euclidean_distance_f32(v1: &[f32], v2: &[f32]) -> f32 {
    if v1.len() != v2.len() {
        panic!("Vectors must be of the same length");
    }
    v1.iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

pub fn dot_product_bf16(v1: &[bf16], v2: &[bf16]) -> f32 {
    if v1.len() != v2.len() {
        panic!("Vectors must be of the same length");
    }
    let dp: f32 = bf16::to_f32(v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum::<bf16>());
    dp * -1.0
}

pub fn euclidean_distance_bf16(v1: &[bf16], v2: &[bf16]) -> f32 {
    if v1.len() != v2.len() {
        panic!("Vectors must be of the same length");
    }
    bf16::to_f32(
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<bf16>(),
    )
    .sqrt()
}

pub fn hamming_distance(v1: &[u8], v2: &[u8]) -> f32 {
    if v1.len() != v2.len() {
        panic!("Vectors must be of the same length");
    }
    v1.iter()
        .zip(v2.iter())
        .map(|(a, b)| (a ^ b).count_ones())
        .sum::<u32>() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_f32() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];
        assert_eq!(dot_product_f32(&v1, &v2), -32.0);
    }

    #[test]
    fn test_euclidean_distance_f32() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];
        assert_eq!(euclidean_distance_f32(&v1, &v2), 5.196152422706632);
    }

    #[test]
    fn test_dot_product_bf16() {
        let v1 = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
        ];
        let v2 = vec![
            bf16::from_f32(4.0),
            bf16::from_f32(5.0),
            bf16::from_f32(6.0),
        ];
        assert_eq!(dot_product_bf16(&v1, &v2), -32.0);
    }

    #[test]
    fn test_euclidean_distance_bf16() {
        let v1 = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
        ];
        let v2 = vec![
            bf16::from_f32(4.0),
            bf16::from_f32(5.0),
            bf16::from_f32(6.0),
        ];
        assert_eq!(euclidean_distance_bf16(&v1, &v2), 5.196152422706632);
    }

    #[test]
    fn test_hamming_distance() {
        let v1 = vec![0b11001100, 0b10101010];
        let v2 = vec![0b11010010, 0b10111111];
        assert_eq!(hamming_distance(&v1, &v2), 7.0);
    }
}
