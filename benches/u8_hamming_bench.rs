use criterion::{criterion_group, criterion_main, Criterion};
use mini_vs::{DistanceMetric, KNNIndex, Vector, VectorType};
use rand::prelude::*;

const DIM: usize = 48;

fn generate_random_vector(dim: usize) -> Vec<u8> {
    let mut rng = rand::rng();
    (0..dim)
        .map(|_| rng.random_range(0..std::u8::MAX))
        .collect()
}

fn bench_knn_index_u8_hamming(c: &mut Criterion) {
    let mut data = Vec::new();
    for _ in 0..10000 {
        let vec: Vec<u8> = generate_random_vector(DIM);
        data.push(vec);
    }

    c.bench_function("KNNIndex_u8_Hamming_add", |b| {
        b.iter(|| {
            let mut index = KNNIndex::new(
                DIM,
                "test_index".to_string(),
                DistanceMetric::Hamming,
                VectorType::Binary,
            );
            for (i, v) in data.iter().enumerate() {
                index.add(&i.to_string(), Vector::from_vec_u8(v.clone()));
            }
        })
    });

    let mut index = KNNIndex::new(
        DIM,
        "test_index".to_string(),
        DistanceMetric::Hamming,
        VectorType::Binary,
    );
    for (i, v) in data.iter().enumerate() {
        index.add(&i.to_string(), Vector::from_vec_u8(v.clone()));
    }

    c.bench_function("KNNIndex_u8_Hamming_search", |b| {
        b.iter(|| {
            for _ in 0..100 {
                index.search(Vector::from_vec_u8(generate_random_vector(DIM)), 100);
            }
        })
    });

    c.bench_function("KNNIndex_u8_remove_data", |b| {
        b.iter(|| {
            let mut index = KNNIndex::new(
                DIM,
                "test_index".to_string(),
                DistanceMetric::Hamming,
                VectorType::Binary,
            );
            for (i, v) in data.iter().enumerate() {
                index.add(&i.to_string(), Vector::from_vec_u8(v.clone()));
            }
            for i in 0..1000 {
                let removed = index.remove(&i.to_string());
                if removed.is_none() {
                    panic!("Failed to remove item with id {}", i);
                }
            }
        })
    });
}

criterion_group!(benches, bench_knn_index_u8_hamming);
criterion_main!(benches);
