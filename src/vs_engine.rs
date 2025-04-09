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
use crate::vs_engine::vector_types::VectorKind;

/// KNNIndex is a simple K-Nearest Neighbors index that supports adding, removing,
/// and searching for vectors.
/// It uses a hash map to store the vectors and their corresponding IDs.
/// The index performs search via an exact (exhaustive) search algorithm.
/// Example usage:
/// ```
/// use mini_vs::{KNNIndex, DistanceMetric, F32Vector};
///
/// fn main() {
///    let mut index: KNNIndex<String, F32Vector> = KNNIndex::new(3, "example_index".to_string(), DistanceMetric::Euclidean);
///    index.add(&"1".to_string(), vec![1.0_f32, 2.0, 3.0]);
///    index.add(&"2".to_string(), vec![4.0_f32, 5.0, 6.0]);
///    index.add(&"3".to_string(), vec![7.0_f32, 8.0, 9.0]);
///
///    let query = vec![1.0_f32, 2.0, 3.0];
///
///    let results = index.search(query, 2);
///    for hit in results {
///        println!("ID: {}, Similarity: {}", hit.id, hit.similarity);
///    }
///  }
/// ```
pub struct KNNIndex<K: Eq + Hash + Clone + Sync + Display, V: VectorKind> {
    data: Vec<Vec<V::Elem>>,
    dim: usize,
    id_to_idx: HashMap<K, usize>,
    idx_to_id: HashMap<usize, K>,
    index_name: String,
    metric: distance::DistanceMetric,
}

// Core index operations
impl<K, V> KNNIndex<K, V>
where
    K: Eq + Hash + Clone + Sync + Display,
    V: VectorKind,
    V::Elem: Sync,
{
    /// Creates a new KNNIndex with the specified dimension, index name, and distance metric.
    /// The dimension must match the dimension of the vectors that will be added to the index.
    /// The index name is used for identification purposes.
    pub fn new(dim: usize, index_name: String, metric: distance::DistanceMetric) -> Self {
        KNNIndex {
            data: Vec::new(),
            dim,
            index_name,
            id_to_idx: HashMap::new(),
            idx_to_id: HashMap::new(),
            metric,
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
    pub fn contains_id(&self, id: &K) -> bool {
        self.id_to_idx.contains_key(id)
    }

    /// Get the vector associated with the given ID.
    pub fn get_vector(&self, id: &K) -> Option<&Vec<V::Elem>> {
        let idx = self.id_to_idx.get(id)?;
        Some(&self.data[*idx])
    }

    /// Try and get all vectors for all hits, if any vector cannot be found then an an error is returned.
    pub fn try_get_hit_vectors(
        &self,
        hits: &Vec<Hit<K>>,
    ) -> Result<Vec<Vec<V::Elem>>, Box<dyn Error>> {
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
    pub fn get_hit_vectors(&self, hits: &Vec<Hit<K>>) -> Vec<Option<Vec<V::Elem>>> {
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
    fn compute_distance(&self, v1: &Vec<V::Elem>, v2: &Vec<V::Elem>) -> f32 {
        V::distance(v1, v2, self.metric)
    }

    /// Add a new vector into the index with the specified ID.
    pub fn add(&mut self, id: &K, vector: Vec<V::Elem>) {
        if vector.len() != self.dim {
            panic!("Vector dimension mismatch.");
        }

        let new_idx = self.data.len();
        self.data.push(vector);

        self.id_to_idx.insert(id.clone(), new_idx);
        self.idx_to_id.insert(new_idx, id.clone());
    }

    /// Remove a vector from the index by its ID.
    pub fn remove(&mut self, id: &K) -> Option<K> {
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
    pub fn search(&self, query: Vec<V::Elem>, k: usize) -> Vec<Hit<K>> {
        if query.len() != self.dim {
            panic!("Query dimension mismatch.");
        }

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

        let hits: Vec<Hit<K>> = heap.into_sorted_vec();

        hits
    }
}

// ser der operations for the index
// impl<K, V> KNNIndex<K, V>
// where
//     K: Eq + Hash + Clone + Sync + Display,
//     V: VectorKind,
//     V::Elem: Copy + Send + Sync,
// {
//     /// Write the index to disk in a binary format.
//     pub fn write_to_disk<W: Write>(&self, writer: &mut W) -> io::Result<()> {
//         writer.write_all(&(self.dim as u64).to_le_bytes())?;
//         writer.write_all(&(self.data.len() as u64).to_le_bytes())?;
//         writer.write_all(&(self.metric as u8).to_le_bytes())?;

//         // Serialize index_name
//         writer.write_all(&(self.index_name.len() as u64).to_le_bytes())?;
//         writer.write_all(self.index_name.as_bytes())?;

//         // Serialize data
//         for vector in &self.data {
//             writer.write_all(&V::to_le_bytes(vector))?;
//         }

//         // Serialize id_to_idx
//         writer.write_all(&(self.id_to_idx.len() as u64).to_le_bytes())?;
//         for (id, idx) in &self.id_to_idx {
//             let id_str = id.to_string();
//             writer.write_all(&(id_str.len() as u64).to_le_bytes())?;
//             writer.write_all(id_str.as_bytes())?;
//             writer.write_all(&(*idx as u64).to_le_bytes())?;
//         }

//         Ok(())
//     }

//     /// Read the index from disk in a binary format.
//     pub fn read_from_disk<R: Read>(reader: &mut R) -> io::Result<Self> {
//         fn read_exact<const N: usize, R: Read>(reader: &mut R) -> io::Result<[u8; N]> {
//             let mut buf = [0u8; N];
//             reader.read_exact(&mut buf)?;
//             Ok(buf)
//         }

//         let dim = u64::from_le_bytes(read_exact::<8, _>(reader)?) as usize;
//         let data_len = u64::from_le_bytes(read_exact::<8, _>(reader)?) as usize;

//         let metric = match u8::from_le_bytes(read_exact::<1, _>(reader)?) {
//             0 => distance::DistanceMetric::Euclidean,
//             1 => distance::DistanceMetric::DotProduct,
//             2 => distance::DistanceMetric::Hamming,
//             x => panic!("Invalid DistanceMetric: {}", x),
//         };

//         // Deserialize index_name
//         let name_len = u64::from_le_bytes(read_exact::<8, _>(reader)?) as usize;
//         let mut name_buf = vec![0u8; name_len];
//         reader.read_exact(&mut name_buf)?;
//         let index_name = String::from_utf8(name_buf).unwrap();

//         // Deserialize data
//         let mut data = Vec::with_capacity(data_len);
//         for _ in 0..data_len {
//             let mut vec_buf = vec![0u8; V::size_of_vector(dim)];
//             reader.read_exact(&mut vec_buf)?;
//             let vector = V::from_le_bytes(&vec_buf, dim);
//             data.push(vector);
//         }

//         // Deserialize id_to_idx
//         let map_len = u64::from_le_bytes(read_exact::<8, _>(reader)?) as usize;
//         let mut id_to_idx = HashMap::with_capacity(map_len);
//         let mut idx_to_id = HashMap::with_capacity(map_len);
//         for _ in 0..map_len {
//             let id_len = u64::from_le_bytes(read_exact::<8, _>(reader)?) as usize;
//             let mut id_buf = vec![0u8; id_len];
//             reader.read_exact(&mut id_buf)?;
//             let id_str = String::from_utf8(id_buf).unwrap();
//             let id = K::from_str(&id_str).unwrap();
//             let idx = u64::from_le_bytes(read_exact::<8, _>(reader)?) as usize;
//             id_to_idx.insert(id.clone(), idx);
//             idx_to_id.insert(idx, id);
//         }

//         Ok(KNNIndex {
//             data,
//             dim,
//             id_to_idx,
//             idx_to_id,
//             index_name,
//             metric,
//         })
//     }
// }

#[cfg(test)]
mod tests {
    use crate::{BF16Vector, BinaryVector, F32Vector};

    use super::*;
    use half::bf16;
    use std::fs::File;
    use tempfile::tempdir;

    #[test]
    fn test_knn_index_f32() {
        let mut index: KNNIndex<String, F32Vector> = KNNIndex::new(
            3,
            "test_index".to_string(),
            distance::DistanceMetric::Euclidean,
        );
        index.add(&"1".to_string(), vec![1.0_f32, 2.0, 3.0]);
        index.add(&"2".to_string(), vec![4.0_f32, 5.0, 6.0]);
        index.add(&"3".to_string(), vec![7.0_f32, 8.0, 9.0]);

        assert_eq!(index.get_name(), "test_index");
        assert!(index.contains_id(&"1".to_string()));
        assert!(!index.contains_id(&"4".to_string()));

        let results = index.search(vec![1.0_f32, 2.0, 3.0], 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "1");
    }

    #[test]
    fn test_knn_index_bf16() {
        let mut index: KNNIndex<String, BF16Vector> = KNNIndex::new(
            3,
            "test_index".to_string(),
            distance::DistanceMetric::Euclidean,
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
        index.add(&"1".to_string(), v1);
        index.add(&"2".to_string(), v2);
        index.add(&"3".to_string(), v3);

        assert_eq!(index.get_name(), "test_index");
        assert!(index.contains_id(&"1".to_string()));
        assert!(!index.contains_id(&"4".to_string()));

        let query = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
        ];

        let results = index.search(query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "1");
    }

    #[test]
    fn test_knn_index_binary() {
        let mut index: KNNIndex<String, BinaryVector> = KNNIndex::new(
            3,
            "test_index".to_string(),
            distance::DistanceMetric::Hamming,
        );

        index.add(&"1".to_string(), vec![128_u8, 128, 128]);
        index.add(&"2".to_string(), vec![127_u8, 127, 127]);
        index.add(&"3".to_string(), vec![4_u8, 4, 4]);

        assert_eq!(index.get_name(), "test_index");
        assert!(index.contains_id(&"1".to_string()));
        assert!(!index.contains_id(&"4".to_string()));

        let results = index.search(vec![128_u8, 128, 128], 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "1");
        assert_eq!(results[1].id, "3");
        assert_eq!(results[0].similarity, 0.0);
        assert_eq!(results[1].similarity, 6.0);
    }

    #[test]
    fn test_knn_index_remove() {
        let mut index: KNNIndex<String, F32Vector> = KNNIndex::new(
            3,
            "test_index".to_string(),
            distance::DistanceMetric::Euclidean,
        );
        index.add(&"1".to_string(), vec![1.0_f32, 2.0, 3.0]);
        index.add(&"2".to_string(), vec![4.0_f32, 5.0, 6.0]);
        index.add(&"3".to_string(), vec![7.0_f32, 8.0, 9.0]);

        assert!(index.contains_id(&"1".to_string()));
        assert!(index.remove(&"1".to_string()).is_some());
        assert!(!index.contains_id(&"1".to_string()));
    }

    #[test]
    fn test_get_vector() {
        let mut index: KNNIndex<String, F32Vector> = KNNIndex::new(
            3,
            "test_index".to_string(),
            distance::DistanceMetric::Euclidean,
        );
        index.add(&"1".to_string(), vec![1.0_f32, 2.0, 3.0]);
        index.add(&"2".to_string(), vec![4.0_f32, 5.0, 6.0]);
        index.add(&"3".to_string(), vec![7.0_f32, 8.0, 9.0]);

        assert!(index.get_vector(&"1".to_string()).is_some());
        assert!(index.get_vector(&"4".to_string()).is_none());

        let v = index.get_vector(&"1".to_string()).unwrap();
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);
    }

    // #[test]
    // fn test_write_read_index() {
    //     let dir = tempdir().unwrap();
    //     let path = dir.path().join("test_index.bin");

    //     let mut index = KNNIndex::new(
    //         3,
    //         "test_index".to_string(),
    //         distance::DistanceMetric::Euclidean,
    //         VectorType::Float,
    //     );
    //     index.add(&"1".to_string(), Vector::from_vec_f32(vec![1.0, 2.0, 3.0]));
    //     index.add(&"2".to_string(), Vector::from_vec_f32(vec![4.0, 5.0, 6.0]));
    //     index.add(&"3".to_string(), Vector::from_vec_f32(vec![7.0, 8.0, 9.0]));

    //     {
    //         let mut file = File::create(&path).unwrap();
    //         index.write_to_disk(&mut file).unwrap();
    //     }

    //     let loaded_index =
    //         KNNIndex::<String>::read_from_disk(&mut File::open(&path).unwrap()).unwrap();

    //     assert_eq!(index.get_name(), loaded_index.get_name());
    //     assert_eq!(index.dim, loaded_index.dim);
    //     assert_eq!(index.metric, loaded_index.metric);
    //     assert_eq!(index.vector_type, loaded_index.vector_type);

    //     assert_eq!(index.data.len(), loaded_index.data.len());
    // }
}
