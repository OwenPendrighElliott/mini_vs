use std::cmp::{Ord, Ordering, PartialOrd};

/// Represents a hit in the KNN search.
/// Contains the ID of the vector and its similarity score.
#[derive(Debug, Clone)]
pub struct Hit<T> {
    pub id: T,
    pub similarity: f32,
}

impl<T> PartialEq for Hit<T> {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity
    }
}
impl<T> Eq for Hit<T> {}

impl<T> PartialOrd for Hit<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.similarity.partial_cmp(&other.similarity)
    }
}
impl<T> Ord for Hit<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or_else(|| {
            if self.similarity.is_nan() && other.similarity.is_nan() {
                Ordering::Equal
            } else if self.similarity.is_nan() {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hit_compare() {
        let hit1 = Hit {
            id: "1",
            similarity: 0.5,
        };
        let hit2 = Hit {
            id: "2",
            similarity: 0.7,
        };
        assert!(hit1 < hit2);
    }
}
