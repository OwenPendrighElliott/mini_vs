use std::cmp::{Ord, Ordering, PartialOrd};
use std::fmt;
use std::fmt::Display;

/// Represents a hit in the KNN search.
#[derive(Debug, Clone)]
pub struct Hit<T: Display> {
    pub id: T,
    pub similarity: f32,
}

impl<T: Display> PartialEq for Hit<T> {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity
    }
}
impl<T: Display> Eq for Hit<T> {}

impl<T: Display> PartialOrd for Hit<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.similarity.partial_cmp(&other.similarity)
    }
}
impl<T: Display> Ord for Hit<T> {
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

impl<T: Display> Display for Hit<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Hit {{ id: {}, similarity: {} }}",
            self.id, self.similarity
        )
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
