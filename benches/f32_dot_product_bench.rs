use criterion::{criterion_group, criterion_main, Criterion};
use mini_vs::{DistanceMetric, KNNIndex, Vector, VectorType};
use rand::prelude::*;

const DIM: usize = 384;

fn generate_random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    (0..dim).map(|_| rng.random_range(0.0..1.0)).collect()
}

fn bench_knn_index_f32_dot_product(c: &mut Criterion) {
    let mut data = Vec::new();
    for _ in 0..10000 {
        let vec: Vec<f32> = generate_random_vector(DIM);
        data.push(vec);
    }

    c.bench_function("KNNIndex_f32_dotproduct_add", |b| {
        b.iter(|| {
            let mut index = KNNIndex::new(
                DIM,
                "test_index".to_string(),
                DistanceMetric::DotProduct,
                VectorType::Float,
            );
            for (i, v) in data.iter().enumerate() {
                index.add(&i.to_string(), Vector::from_vec_f32(v.clone()));
            }
        })
    });

    let mut index = KNNIndex::new(
        DIM,
        "test_index".to_string(),
        DistanceMetric::DotProduct,
        VectorType::Float,
    );
    for (i, v) in data.iter().enumerate() {
        index.add(&i.to_string(), Vector::from_vec_f32(v.clone()));
    }

    c.bench_function("KNNIndex_f32_dotproduct_search", |b| {
        b.iter(|| {
            for _ in 0..100 {
                index.search(Vector::from_vec_f32(generate_random_vector(DIM)), 100);
            }
        })
    });

    c.bench_function("KNNIndex_f32_remove_data", |b| {
        b.iter(|| {
            let mut index = KNNIndex::new(
                DIM,
                "test_index".to_string(),
                DistanceMetric::DotProduct,
                VectorType::Float,
            );
            for (i, v) in data.iter().enumerate() {
                index.add(&i.to_string(), Vector::from_vec_f32(v.clone()));
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

criterion_group!(benches, bench_knn_index_f32_dot_product);
criterion_main!(benches);
