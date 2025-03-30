use half::bf16;

/// Enum to represent different vector types.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum VectorType {
    Float,
    Binary,
    BF16,
}

/// Enum to represent different vector types and their data.
pub enum Vector {
    Float(Vec<f32>),
    Binary(Vec<u8>),
    BF16(Vec<bf16>),
}

impl Vector {
    /// Creates a new vector of the specified type (f32).
    pub fn from_vec_f32(vec: Vec<f32>) -> Self {
        Vector::Float(vec)
    }

    /// Creates a new vector of the specified type (bf16).
    pub fn from_vec_bf16(vec: Vec<bf16>) -> Self {
        Vector::BF16(vec)
    }

    /// Creates a new vector of the specified type (u8).
    /// This is typically used for binary vectors.
    pub fn from_vec_u8(vec: Vec<u8>) -> Self {
        Vector::Binary(vec)
    }

    /// Returns the length of the vector.
    pub fn len(&self) -> usize {
        match self {
            Vector::Float(v) => v.len(),
            Vector::Binary(v) => v.len(),
            Vector::BF16(v) => v.len(),
        }
    }

    /// Returns the actual dimension of the vector, for f32 and bf16 it is the length but for binary it is the number of bits.
    pub fn get_actual_dim(&self) -> usize {
        match self {
            Vector::Float(v) => v.len(),
            Vector::Binary(v) => v.len() * 8,
            Vector::BF16(v) => v.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_len() {
        let vec_f32 = Vector::from_vec_f32(vec![1.0, 2.0, 3.0]);
        assert_eq!(vec_f32.len(), 3);

        let vec_bf16 = Vector::from_vec_bf16(vec![bf16::from_f32(1.0), bf16::from_f32(2.0)]);
        assert_eq!(vec_bf16.len(), 2);

        let vec_u8 = Vector::from_vec_u8(vec![1, 2, 3]);
        assert_eq!(vec_u8.len(), 3);
    }

    #[test]
    fn test_vector_get_actual_dim() {
        let vec_f32 = Vector::from_vec_f32(vec![1.0, 2.0, 3.0]);
        assert_eq!(vec_f32.get_actual_dim(), 3);

        let vec_bf16 = Vector::from_vec_bf16(vec![bf16::from_f32(1.0), bf16::from_f32(2.0)]);
        assert_eq!(vec_bf16.get_actual_dim(), 2);

        let vec_u8 = Vector::from_vec_u8(vec![1, 2, 3]);
        assert_eq!(vec_u8.get_actual_dim(), 24);
    }
}
