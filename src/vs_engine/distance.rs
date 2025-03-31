use crate::vs_engine::vector_types::Vector;
use half::bf16;

/// Enum to represent different distance metrics.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DistanceMetric {
    Euclidean = 0,
    DotProduct = 1,
    Hamming = 2,
}

/// Calculate the distance between two vectors based on the specified metric.
pub fn calculate_distance(v1: &Vector, v2: &Vector, metric: &DistanceMetric) -> f32 {
    match (v1, v2, metric) {
        (Vector::Float(a), Vector::Float(b), DistanceMetric::Euclidean) => {
            euclidean_distance_f32(a, b)
        }
        (Vector::Float(a), Vector::Float(b), DistanceMetric::DotProduct) => dot_product_f32(a, b),
        (Vector::BF16(a), Vector::BF16(b), DistanceMetric::Euclidean) => {
            euclidean_distance_bf16(a, b)
        }
        (Vector::BF16(a), Vector::BF16(b), DistanceMetric::DotProduct) => dot_product_bf16(a, b),
        (Vector::Binary(a), Vector::Binary(b), DistanceMetric::Hamming) => hamming_distance(a, b),
        _ => panic!("Invalid combination of vector data type and distance metric!"),
    }
}

pub fn dot_product_f32(v1: &[f32], v2: &[f32]) -> f32 {
    let dp: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    dp * -1.0
}

pub fn euclidean_distance_f32(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

pub fn dot_product_bf16(v1: &[bf16], v2: &[bf16]) -> f32 {
    let dp: f32 = bf16::to_f32(v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum::<bf16>());
    dp * -1.0
}

pub fn euclidean_distance_bf16(v1: &[bf16], v2: &[bf16]) -> f32 {
    bf16::to_f32(
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<bf16>(),
    )
    .sqrt()
}

pub fn hamming_distance(v1: &[u8], v2: &[u8]) -> f32 {
    v1.iter()
        .zip(v2.iter())
        .map(|(a, b)| (a ^ b).count_ones())
        .sum::<u32>() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vs_engine::vector_types::Vector;

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

    #[test]
    fn test_calculate_distance() {
        let v1 = Vector::from_vec_f32(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::from_vec_f32(vec![4.0, 5.0, 6.0]);
        assert_eq!(
            calculate_distance(&v1, &v2, &DistanceMetric::Euclidean),
            5.196152422706632
        );
        assert_eq!(
            calculate_distance(&v1, &v2, &DistanceMetric::DotProduct),
            -32.0
        );

        let v1_bf16 = Vector::from_vec_bf16(vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
        ]);
        let v2_bf16 = Vector::from_vec_bf16(vec![
            bf16::from_f32(4.0),
            bf16::from_f32(5.0),
            bf16::from_f32(6.0),
        ]);
        assert_eq!(
            calculate_distance(&v1_bf16, &v2_bf16, &DistanceMetric::Euclidean),
            5.196152422706632
        );
        assert_eq!(
            calculate_distance(&v1_bf16, &v2_bf16, &DistanceMetric::DotProduct),
            -32.0
        );

        let v1_bin = Vector::from_vec_u8(vec![0b11001100, 0b10101010]);
        let v2_bin = Vector::from_vec_u8(vec![0b11010010, 0b10111111]);
        assert_eq!(
            calculate_distance(&v1_bin, &v2_bin, &DistanceMetric::Hamming),
            7.0
        );
    }
}
