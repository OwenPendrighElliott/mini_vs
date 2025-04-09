use crate::vs_engine::distance::{
    dot_product_bf16, dot_product_f32, euclidean_distance_bf16, euclidean_distance_f32,
    hamming_distance, DistanceMetric,
};
use half::bf16;

pub struct F32Vector;
pub struct BF16Vector;
pub struct BinaryVector;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::F32Vector {}
    impl Sealed for super::BF16Vector {}
    impl Sealed for super::BinaryVector {}
}

pub trait VectorKind: sealed::Sealed {
    type Elem: Copy + Send + Sync;
    fn distance(a: &[Self::Elem], b: &[Self::Elem], metric: DistanceMetric) -> f32;
    fn to_le_bytes(a: &[Self::Elem]) -> Vec<u8>;
    fn from_le_bytes(bytes: &[u8], dim: usize) -> Vec<Self::Elem>;
    fn size_of_vector(dim: usize) -> usize {
        std::mem::size_of::<Self::Elem>() * dim
    }
}

impl VectorKind for F32Vector {
    type Elem = f32;

    fn distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
        match metric {
            DistanceMetric::Euclidean => euclidean_distance_f32(a, b),
            DistanceMetric::DotProduct => dot_product_f32(a, b),
            _ => panic!("Invalid metric for f32 vector"),
        }
    }

    fn to_le_bytes(a: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &value in a {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn from_le_bytes(bytes: &[u8], dim: usize) -> Vec<f32> {
        let mut vec = Vec::with_capacity(dim);
        for chunk in bytes.chunks_exact(std::mem::size_of::<f32>()) {
            let value = f32::from_le_bytes(chunk.try_into().unwrap());
            vec.push(value);
        }
        vec
    }
}

impl VectorKind for BF16Vector {
    type Elem = bf16;

    fn distance(a: &[bf16], b: &[bf16], metric: DistanceMetric) -> f32 {
        match metric {
            DistanceMetric::Euclidean => euclidean_distance_bf16(a, b),
            DistanceMetric::DotProduct => dot_product_bf16(a, b),
            _ => panic!("Invalid metric for bf16 vector"),
        }
    }

    fn to_le_bytes(a: &[bf16]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &value in a {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes
    }

    fn from_le_bytes(bytes: &[u8], dim: usize) -> Vec<bf16> {
        let mut vec = Vec::with_capacity(dim);
        for chunk in bytes.chunks_exact(std::mem::size_of::<bf16>()) {
            let value = bf16::from_ne_bytes(chunk.try_into().unwrap());
            vec.push(value);
        }
        vec
    }
}

impl VectorKind for BinaryVector {
    type Elem = u8;

    fn distance(a: &[u8], b: &[u8], metric: DistanceMetric) -> f32 {
        match metric {
            DistanceMetric::Hamming => hamming_distance(a, b),
            _ => panic!("Invalid metric for binary vector"),
        }
    }

    fn to_le_bytes(a: &[u8]) -> Vec<u8> {
        a.to_vec()
    }
    fn from_le_bytes(bytes: &[u8], _: usize) -> Vec<u8> {
        bytes.to_vec()
    }
}
