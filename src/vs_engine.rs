use std::collections::{BinaryHeap, HashMap};

use rayon::prelude::*;

pub mod distance;
pub mod hit;
pub mod vector_types;
use crate::vs_engine::hit::Hit;
use crate::vs_engine::vector_types::{Vector, VectorType};

/// KNNIndex is a simple K-Nearest Neighbors index that supports adding, removing,
/// and searching for vectors.
/// It uses a hash map to store the vectors and their corresponding IDs.
/// The index performs search via an exact (exhaustive) search algorithm.
/// Example usage:
/// ```
/// use mini_vs::{KNNIndex, Vector, VectorType, DistanceMetric};
///
/// fn main() {
///    let mut index = KNNIndex::new(3, "example_index".to_string(), DistanceMetric::Euclidean, VectorType::Float);
///   index.add("1", Vector::from_vec_f32(vec![1.0, 2.0, 3.0]));
///   index.add("2", Vector::from_vec_f32(vec![4.0, 5.0, 6.0]));
///   index.add("3", Vector::from_vec_f32(vec![7.0, 8.0, 9.0]));
///
///   let query = Vector::from_vec_f32(vec![1.0, 2.0, 3.0]);
///
///   let results = index.search(query, 2);
///   for hit in results {
///       println!("ID: {}, Similarity: {}", hit.id, hit.similarity);
///   }
/// }
/// ```
pub struct KNNIndex {
    data: Vec<Vector>,
    dim: usize,
    id_to_idx: HashMap<String, usize>,
    idx_to_id: HashMap<usize, String>,
    index_name: String,
    metric: distance::DistanceMetric,
    vector_type: VectorType,
}

impl KNNIndex {
    /// Creates a new KNNIndex with the specified dimension, index name, and distance metric.
    /// The dimension must match the dimension of the vectors that will be added to the index.
    /// The index name is used for identification purposes.
    pub fn new(
        dim: usize,
        index_name: String,
        metric: distance::DistanceMetric,
        vector_type: VectorType,
    ) -> Self {
        KNNIndex {
            data: Vec::new(),
            dim,
            index_name,
            id_to_idx: HashMap::new(),
            idx_to_id: HashMap::new(),
            metric,
            vector_type,
        }
    }

    /// Returns the name of the index.
    pub fn get_name(&self) -> &str {
        &self.index_name
    }

    /// Checks if the index contains a vector with the specified ID.
    pub fn contains_id(&self, id: &str) -> bool {
        self.id_to_idx.contains_key(id)
    }

    /// Get the vector associated with the given ID.
    pub fn get_vector(&self, id: &str) -> Option<&Vector> {
        let idx = self.id_to_idx.get(id);
        Some(&self.data[*idx?])
    }

    /// Internal helper to wrap the distance calculation.
    fn compute_distance(&self, v1: &Vector, v2: &Vector) -> f32 {
        distance::calculate_distance(v1, v2, &self.metric)
    }

    /// Check if the vector type matches the expected type for this index.
    fn check_vector_type(&self, vector: &Vector) {
        match (&self.vector_type, vector) {
            (VectorType::Float, Vector::Float(_)) => {}
            (VectorType::Binary, Vector::Binary(_)) => {}
            (VectorType::BF16, Vector::BF16(_)) => {}
            _ => panic!("Mismatched vector type for this store!"),
        }
    }

    /// Add a new vector into the index with the specified ID.
    pub fn add(&mut self, id: &str, vector: Vector) {
        if vector.len() != self.dim {
            panic!("Vector dimension mismatch.");
        }

        self.check_vector_type(&vector);

        let new_idx = self.data.len();
        self.data.push(vector);

        self.id_to_idx.insert(id.to_string(), new_idx);
        self.idx_to_id.insert(new_idx, id.to_string());
    }

    /// Remove a vector from the index by its ID.
    pub fn remove(&mut self, id: &str) -> Option<String> {
        let idx = self.id_to_idx.remove(id);

        if idx? >= self.data.len() {
            panic!("Index out of bounds.");
        }
        let last_idx = self.data.len() - 1;
        let last_id = self.idx_to_id[&last_idx].clone();

        self.data.swap_remove(idx?);
        self.idx_to_id.remove(&last_idx);
        self.idx_to_id.insert(idx?, last_id.to_string());
        self.id_to_idx.insert(last_id.to_string(), idx?);
        Some(id.to_string())
    }

    /// Search for the k nearest neighbors of a given query vector.
    /// Returns a vector of Hit objects containing the IDs and similarities of the nearest neighbors.
    pub fn search(&self, query: Vector, k: usize) -> Vec<Hit<String>> {
        if query.len() != self.dim {
            panic!("Query dimension mismatch.");
        }

        self.check_vector_type(&query);

        let neighbors: Vec<Hit<usize>> = self
            .data
            .par_iter()
            .enumerate()
            .map(|(index, vector)| {
                let similarity = self.compute_distance(&query, vector);
                Hit {
                    id: index,
                    similarity,
                }
            })
            .collect();

        let mut heap = BinaryHeap::new();
        for neighbor in neighbors {
            let id_str = self.idx_to_id.get(&neighbor.id).unwrap().clone();
            heap.push(Hit {
                id: id_str,
                similarity: neighbor.similarity,
            });
            if heap.len() > k {
                heap.pop();
            }
        }

        let hits: Vec<Hit<String>> = heap.into_sorted_vec();

        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    #[test]
    fn test_knn_index_f32() {
        let mut index = KNNIndex::new(
            3,
            "test_index".to_string(),
            distance::DistanceMetric::Euclidean,
            VectorType::Float,
        );
        index.add("1", Vector::from_vec_f32(vec![1.0, 2.0, 3.0]));
        index.add("2", Vector::from_vec_f32(vec![4.0, 5.0, 6.0]));
        index.add("3", Vector::from_vec_f32(vec![7.0, 8.0, 9.0]));

        assert_eq!(index.get_name(), "test_index");
        assert!(index.contains_id("1"));
        assert!(!index.contains_id("4"));

        let results = index.search(Vector::from_vec_f32(vec![1.0, 2.0, 3.0]), 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "1");
    }

    #[test]
    fn test_knn_index_bf16() {
        let mut index = KNNIndex::new(
            3,
            "test_index".to_string(),
            distance::DistanceMetric::Euclidean,
            VectorType::BF16,
        );
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
        let v3 = vec![
            bf16::from_f32(7.0),
            bf16::from_f32(8.0),
            bf16::from_f32(9.0),
        ];
        index.add("1", Vector::from_vec_bf16(v1));
        index.add("2", Vector::from_vec_bf16(v2));
        index.add("3", Vector::from_vec_bf16(v3));

        assert_eq!(index.get_name(), "test_index");
        assert!(index.contains_id("1"));
        assert!(!index.contains_id("4"));

        let query = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
        ];

        let results = index.search(Vector::from_vec_bf16(query), 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "1");
    }

    #[test]
    fn test_knn_index_binary() {
        let mut index = KNNIndex::new(
            3,
            "test_index".to_string(),
            distance::DistanceMetric::Hamming,
            VectorType::Binary,
        );

        index.add("1", Vector::from_vec_u8(vec![128, 128, 128]));
        index.add("2", Vector::from_vec_u8(vec![127, 127, 127]));
        index.add("3", Vector::from_vec_u8(vec![4, 4, 4]));

        assert_eq!(index.get_name(), "test_index");
        assert!(index.contains_id("1"));
        assert!(!index.contains_id("4"));

        let results = index.search(Vector::from_vec_u8(vec![128, 128, 128]), 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "1");
        assert_eq!(results[1].id, "3");
        assert_eq!(results[0].similarity, 0.0);
        assert_eq!(results[1].similarity, 6.0);
    }

    #[test]
    fn test_knn_index_remove() {
        let mut index = KNNIndex::new(
            3,
            "test_index".to_string(),
            distance::DistanceMetric::Euclidean,
            VectorType::Float,
        );
        index.add("1", Vector::from_vec_f32(vec![1.0, 2.0, 3.0]));
        index.add("2", Vector::from_vec_f32(vec![4.0, 5.0, 6.0]));
        index.add("3", Vector::from_vec_f32(vec![7.0, 8.0, 9.0]));

        assert!(index.contains_id("1"));
        assert!(index.remove("1").is_some());
        assert!(!index.contains_id("1"));
    }

    #[test]
    fn test_get_vector() {
        let mut index = KNNIndex::new(
            3,
            "test_index".to_string(),
            distance::DistanceMetric::Euclidean,
            VectorType::Float,
        );
        index.add("1", Vector::from_vec_f32(vec![1.0, 2.0, 3.0]));
        index.add("2", Vector::from_vec_f32(vec![4.0, 5.0, 6.0]));
        index.add("3", Vector::from_vec_f32(vec![7.0, 8.0, 9.0]));

        assert!(index.get_vector(&"1").is_some());
        assert!(index.get_vector(&"4").is_none());

        let vector = index.get_vector("1").unwrap();
        if let Vector::Float(v) = vector {
            assert_eq!(v.len(), 3);
            assert_eq!(v[0], 1.0);
            assert_eq!(v[1], 2.0);
            assert_eq!(v[2], 3.0);
        } else {
            panic!("Expected a Float vector");
        }
    }
}
