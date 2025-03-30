use criterion::{criterion_group, criterion_main, Criterion};
use half::bf16;
use mini_vs::{DistanceMetric, KNNIndex, Vector, VectorType};
use rand::prelude::*;

const DIM: usize = 384;

fn generate_random_vector(dim: usize) -> Vec<bf16> {
    let mut rng = rand::rng();
    let vec: Vec<f32> = (0..dim).map(|_| rng.random_range(0.0..1.0)).collect();

    vec.iter()
        .map(|&x| bf16::from_f32(x))
        .collect::<Vec<bf16>>()
}

fn bench_knn_index_bf16_dot_product(c: &mut Criterion) {
    let mut data = Vec::new();
    for _ in 0..10000 {
        let vec: Vec<bf16> = generate_random_vector(DIM);
        data.push(vec);
    }

    c.bench_function("KNNIndex_bf16_dotproduct_add", |b| {
        b.iter(|| {
            let mut index = KNNIndex::new(
                DIM,
                "test_index".to_string(),
                DistanceMetric::DotProduct,
                VectorType::BF16,
            );
            for (i, v) in data.iter().enumerate() {
                index.add(&i.to_string(), Vector::from_vec_bf16(v.clone()));
            }
        })
    });

    let mut index = KNNIndex::new(
        DIM,
        "test_index".to_string(),
        DistanceMetric::DotProduct,
        VectorType::BF16,
    );
    for (i, v) in data.iter().enumerate() {
        index.add(&i.to_string(), Vector::from_vec_bf16(v.clone()));
    }

    c.bench_function("KNNIndex_bf16_dotproduct_search", |b| {
        b.iter(|| {
            for _ in 0..100 {
                index.search(Vector::from_vec_bf16(generate_random_vector(DIM)), 100);
            }
        })
    });

    c.bench_function("KNNIndex_bf16_remove_data", |b| {
        b.iter(|| {
            let mut index = KNNIndex::new(
                DIM,
                "test_index".to_string(),
                DistanceMetric::DotProduct,
                VectorType::BF16,
            );
            for (i, v) in data.iter().enumerate() {
                index.add(&i.to_string(), Vector::from_vec_bf16(v.clone()));
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

criterion_group!(benches, bench_knn_index_bf16_dot_product);
criterion_main!(benches);
