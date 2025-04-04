use half::bf16;
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashMap};
use std::error::Error;
use std::fmt::Display;
use std::hash::Hash;
use std::io::{self, Read, Write};
use std::str::FromStr;
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
///    index.add(&"1".to_string(), Vector::from_vec_f32(vec![1.0, 2.0, 3.0]));
///    index.add(&"2".to_string(), Vector::from_vec_f32(vec![4.0, 5.0, 6.0]));
///    index.add(&"3".to_string(), Vector::from_vec_f32(vec![7.0, 8.0, 9.0]));
///
///    let query = Vector::from_vec_f32(vec![1.0, 2.0, 3.0]);
///
///    let results = index.search(query, 2);
///    for hit in results {
///        println!("ID: {}, Similarity: {}", hit.id, hit.similarity);
///    }
///  }
/// ```
pub struct KNNIndex<T: Eq + Hash + Clone + Sync + Display> {
    data: Vec<Vector>,
    dim: usize,
    id_to_idx: HashMap<T, usize>,
    idx_to_id: HashMap<usize, T>,
    index_name: String,
    metric: distance::DistanceMetric,
    vector_type: VectorType,
}

// Core index operations
impl<T: Eq + Hash + Clone + Sync + Display> KNNIndex<T> {
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

    /// Reserves space for additional vectors in the index.
    /// This is useful for optimizing memory allocation when you know in advance
    /// how many vectors you will be adding.
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
        self.id_to_idx.reserve(additional);
        self.idx_to_id.reserve(additional);
    }

    /// Returns the name of the index.
    pub fn get_name(&self) -> &str {
        &self.index_name
    }

    /// Checks if the index contains a vector with the specified ID.
    pub fn contains_id(&self, id: &T) -> bool {
        self.id_to_idx.contains_key(id)
    }

    /// Get the vector associated with the given ID.
    pub fn get_vector(&self, id: &T) -> Option<&Vector> {
        let idx = self.id_to_idx.get(id)?;
        Some(&self.data[*idx])
    }

    /// Try and get all vectors for all hits, if any vector cannot be found then an an error is returned.
    pub fn try_get_hit_vectors(&self, hits: &Vec<Hit<T>>) -> Result<Vec<Vector>, Box<dyn Error>> {
        let mut res = Vec::with_capacity(hits.len());
        for h in hits {
            let vector = self.get_vector(&h.id);
            match vector {
                Some(v) => res.push(v.clone()),
                None => return Err(format!("Vector not found for hit {}", h).into()),
            }
        }
        Ok(res)
    }

    /// Get vectors for hits, if any vectors cannot be found they will be None.
    pub fn get_hit_vectors(&self, hits: &Vec<Hit<T>>) -> Vec<Option<Vector>> {
        let mut res = Vec::with_capacity(hits.len());
        for h in hits {
            let vector = self.get_vector(&h.id);
            if let Some(v) = vector {
                res.push(Some(v.clone()));
            } else {
                res.push(None);
            }
        }
        res
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
    pub fn add(&mut self, id: &T, vector: Vector) {
        if vector.len() != self.dim {
            panic!("Vector dimension mismatch.");
        }

        self.check_vector_type(&vector);

        let new_idx = self.data.len();
        self.data.push(vector);

        self.id_to_idx.insert(id.clone(), new_idx);
        self.idx_to_id.insert(new_idx, id.clone());
    }

    /// Remove a vector from the index by its ID.
    pub fn remove(&mut self, id: &T) -> Option<T> {
        let idx = self.id_to_idx.remove(id);

        if idx? >= self.data.len() {
            panic!("Index out of bounds.");
        }
        let last_idx = self.data.len() - 1;
        let last_id = self.idx_to_id[&last_idx].clone();

        self.data.swap_remove(idx?);
        self.idx_to_id.remove(&last_idx);
        self.idx_to_id.insert(idx?, last_id.clone());
        self.id_to_idx.insert(last_id, idx?);
        Some(id.clone())
    }

    /// Search for the k nearest neighbors of a given query vector.
    /// Returns a vector of Hit objects containing the IDs and similarities of the nearest neighbors.
    pub fn search(&self, query: Vector, k: usize) -> Vec<Hit<T>> {
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

        let hits: Vec<Hit<T>> = heap.into_sorted_vec();

        hits
    }
}

// ser der operations for the index
impl<T> KNNIndex<T>
where
    T: Eq + Hash + Clone + Sync + Display + FromStr,
    <T as FromStr>::Err: std::fmt::Debug,
{
    /// Write the index to disk in a binary format.
    pub fn write_to_disk<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&(self.dim as u64).to_le_bytes())?;
        writer.write_all(&(self.data.len() as u64).to_le_bytes())?;
        writer.write_all(&(self.vector_type as u8).to_le_bytes())?;
        writer.write_all(&(self.metric as u8).to_le_bytes())?;

        // Serialize index_name
        writer.write_all(&(self.index_name.len() as u64).to_le_bytes())?;
        writer.write_all(self.index_name.as_bytes())?;

        // Serialize data
        for vector in &self.data {
            match vector {
                Vector::Float(vec) => {
                    for &val in vec {
                        writer.write_all(&val.to_le_bytes())?;
                    }
                }
                Vector::Binary(vec) => {
                    writer.write_all(vec)?;
                }
                Vector::BF16(vec) => {
                    for &val in vec {
                        writer.write_all(&val.to_le_bytes())?;
                    }
                }
            }
        }

        // Serialize id_to_idx
        writer.write_all(&(self.id_to_idx.len() as u64).to_le_bytes())?;
        for (id, idx) in &self.id_to_idx {
            let id_str = id.to_string();
            writer.write_all(&(id_str.len() as u64).to_le_bytes())?;
            writer.write_all(id_str.as_bytes())?;
            writer.write_all(&(*idx as u64).to_le_bytes())?;
        }

        Ok(())
    }

    /// Read the index from disk in a binary format.
    pub fn read_from_disk<R: Read>(reader: &mut R) -> io::Result<Self> {
        fn read_exact<const N: usize, R: Read>(reader: &mut R) -> io::Result<[u8; N]> {
            let mut buf = [0u8; N];
            reader.read_exact(&mut buf)?;
            Ok(buf)
        }

        let dim = u64::from_le_bytes(read_exact::<8, _>(reader)?) as usize;
        let data_len = u64::from_le_bytes(read_exact::<8, _>(reader)?) as usize;
        let vector_type = match u8::from_le_bytes(read_exact::<1, _>(reader)?) {
            0 => VectorType::Float,
            1 => VectorType::Binary,
            2 => VectorType::BF16,
            x => panic!("Invalid VectorType: {}", x),
        };
        let metric = match u8::from_le_bytes(read_exact::<1, _>(reader)?) {
            0 => distance::DistanceMetric::Euclidean,
            1 => distance::DistanceMetric::DotProduct,
            2 => distance::DistanceMetric::Hamming,
            x => panic!("Invalid DistanceMetric: {}", x),
        };

        // Deserialize index_name
        let name_len = u64::from_le_bytes(read_exact::<8, _>(reader)?) as usize;
        let mut name_buf = vec![0u8; name_len];
        reader.read_exact(&mut name_buf)?;
        let index_name = String::from_utf8(name_buf).unwrap();

        // Deserialize data
        let mut data = Vec::with_capacity(data_len);
        for _ in 0..data_len {
            let vec = match vector_type {
                VectorType::Float => {
                    let mut vals = vec![0f32; dim];
                    for val in &mut vals {
                        *val = f32::from_le_bytes(read_exact::<4, _>(reader)?);
                    }
                    Vector::Float(vals)
                }
                VectorType::Binary => {
                    let mut vals = vec![0u8; dim];
                    reader.read_exact(&mut vals)?;
                    Vector::Binary(vals)
                }
                VectorType::BF16 => {
                    let mut vals = vec![bf16::from_f32(0.0); dim];
                    for val in &mut vals {
                        *val = bf16::from_le_bytes(read_exact::<2, _>(reader)?);
                    }
                    Vector::BF16(vals)
                }
            };
            data.push(vec);
        }

        // Deserialize id_to_idx
        let map_len = u64::from_le_bytes(read_exact::<8, _>(reader)?) as usize;
        let mut id_to_idx = HashMap::with_capacity(map_len);
        let mut idx_to_id = HashMap::with_capacity(map_len);
        for _ in 0..map_len {
            let id_len = u64::from_le_bytes(read_exact::<8, _>(reader)?) as usize;
            let mut id_buf = vec![0u8; id_len];
            reader.read_exact(&mut id_buf)?;
            let id_str = String::from_utf8(id_buf).unwrap();
            let id = T::from_str(&id_str).unwrap();
            let idx = u64::from_le_bytes(read_exact::<8, _>(reader)?) as usize;
            id_to_idx.insert(id.clone(), idx);
            idx_to_id.insert(idx, id);
        }

        Ok(KNNIndex {
            data,
            dim,
            id_to_idx,
            idx_to_id,
            index_name,
            metric,
            vector_type,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;
    use std::fs::File;
    use tempfile::tempdir;

    #[test]
    fn test_knn_index_f32() {
        let mut index: KNNIndex<String> = KNNIndex::new(
            3,
            "test_index".to_string(),
            distance::DistanceMetric::Euclidean,
            VectorType::Float,
        );
        index.add(&"1".to_string(), Vector::from_vec_f32(vec![1.0, 2.0, 3.0]));
        index.add(&"2".to_string(), Vector::from_vec_f32(vec![4.0, 5.0, 6.0]));
        index.add(&"3".to_string(), Vector::from_vec_f32(vec![7.0, 8.0, 9.0]));

        assert_eq!(index.get_name(), "test_index");
        assert!(index.contains_id(&"1".to_string()));
        assert!(!index.contains_id(&"4".to_string()));

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
        index.add(&"1".to_string(), Vector::from_vec_bf16(v1));
        index.add(&"2".to_string(), Vector::from_vec_bf16(v2));
        index.add(&"3".to_string(), Vector::from_vec_bf16(v3));

        assert_eq!(index.get_name(), "test_index");
        assert!(index.contains_id(&"1".to_string()));
        assert!(!index.contains_id(&"4".to_string()));

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

        index.add(&"1".to_string(), Vector::from_vec_u8(vec![128, 128, 128]));
        index.add(&"2".to_string(), Vector::from_vec_u8(vec![127, 127, 127]));
        index.add(&"3".to_string(), Vector::from_vec_u8(vec![4, 4, 4]));

        assert_eq!(index.get_name(), "test_index");
        assert!(index.contains_id(&"1".to_string()));
        assert!(!index.contains_id(&"4".to_string()));

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
        index.add(&"1".to_string(), Vector::from_vec_f32(vec![1.0, 2.0, 3.0]));
        index.add(&"2".to_string(), Vector::from_vec_f32(vec![4.0, 5.0, 6.0]));
        index.add(&"3".to_string(), Vector::from_vec_f32(vec![7.0, 8.0, 9.0]));

        assert!(index.contains_id(&"1".to_string()));
        assert!(index.remove(&"1".to_string()).is_some());
        assert!(!index.contains_id(&"1".to_string()));
    }

    #[test]
    fn test_get_vector() {
        let mut index = KNNIndex::new(
            3,
            "test_index".to_string(),
            distance::DistanceMetric::Euclidean,
            VectorType::Float,
        );
        index.add(&"1".to_string(), Vector::from_vec_f32(vec![1.0, 2.0, 3.0]));
        index.add(&"2".to_string(), Vector::from_vec_f32(vec![4.0, 5.0, 6.0]));
        index.add(&"3".to_string(), Vector::from_vec_f32(vec![7.0, 8.0, 9.0]));

        assert!(index.get_vector(&"1".to_string()).is_some());
        assert!(index.get_vector(&"4".to_string()).is_none());

        let vector = index.get_vector(&"1".to_string()).unwrap();
        if let Vector::Float(v) = vector {
            assert_eq!(v.len(), 3);
            assert_eq!(v[0], 1.0);
            assert_eq!(v[1], 2.0);
            assert_eq!(v[2], 3.0);
        } else {
            panic!("Expected a Float vector");
        }
    }

    #[test]
    fn test_write_read_index() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_index.bin");

        let mut index = KNNIndex::new(
            3,
            "test_index".to_string(),
            distance::DistanceMetric::Euclidean,
            VectorType::Float,
        );
        index.add(&"1".to_string(), Vector::from_vec_f32(vec![1.0, 2.0, 3.0]));
        index.add(&"2".to_string(), Vector::from_vec_f32(vec![4.0, 5.0, 6.0]));
        index.add(&"3".to_string(), Vector::from_vec_f32(vec![7.0, 8.0, 9.0]));

        {
            let mut file = File::create(&path).unwrap();
            index.write_to_disk(&mut file).unwrap();
        }

        let loaded_index =
            KNNIndex::<String>::read_from_disk(&mut File::open(&path).unwrap()).unwrap();

        assert_eq!(index.get_name(), loaded_index.get_name());
        assert_eq!(index.dim, loaded_index.dim);
        assert_eq!(index.metric, loaded_index.metric);
        assert_eq!(index.vector_type, loaded_index.vector_type);

        assert_eq!(index.data.len(), loaded_index.data.len());
    }
}
