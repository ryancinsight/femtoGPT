use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use femto_gpt::tensor::Tensor;
use rand::prelude::*;

fn create_random_tensor(shape: &[usize], rng: &mut impl Rng) -> Vec<f32> {
    let size = shape.iter().product();
    (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn tensor_creation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");
    
    let sizes = vec![100, 500, 1000, 2000];
    
    for size in sizes {
        group.bench_with_input(BenchmarkId::new("zeros", size), &size, |b, &size| {
            b.iter(|| {
                let shape = vec![size, size];
                black_box(Tensor::<f32>::zeros(&shape))
            })
        });
        
        group.bench_with_input(BenchmarkId::new("ones", size), &size, |b, &size| {
            b.iter(|| {
                let shape = vec![size, size];
                black_box(Tensor::<f32>::ones(&shape))
            })
        });
        
        group.bench_with_input(BenchmarkId::new("randn", size), &size, |b, &size| {
            let mut rng = StdRng::seed_from_u64(42);
            b.iter(|| {
                let shape = vec![size, size];
                black_box(Tensor::<f32>::randn(&mut rng, &shape))
            })
        });
    }
    
    group.finish();
}

fn tensor_operations_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_operations");
    
    let sizes = vec![100, 500, 1000];
    let mut rng = StdRng::seed_from_u64(42);
    
    for size in sizes {
        let shape = vec![size, size];
        let data1 = create_random_tensor(&shape, &mut rng);
        let data2 = create_random_tensor(&shape, &mut rng);
        let tensor1 = Tensor::new(data1, shape.clone()).unwrap();
        let tensor2 = Tensor::new(data2, shape.clone()).unwrap();
        
        group.bench_with_input(BenchmarkId::new("addition", size), &size, |b, _| {
            b.iter(|| {
                black_box((&tensor1 + &tensor2).unwrap())
            })
        });
        
        group.bench_with_input(BenchmarkId::new("subtraction", size), &size, |b, _| {
            b.iter(|| {
                black_box((&tensor1 - &tensor2).unwrap())
            })
        });
        
        group.bench_with_input(BenchmarkId::new("multiplication", size), &size, |b, _| {
            b.iter(|| {
                black_box((&tensor1 * &tensor2).unwrap())
            })
        });
        
        group.bench_with_input(BenchmarkId::new("scalar_multiplication", size), &size, |b, _| {
            b.iter(|| {
                black_box(&tensor1 * 2.0)
            })
        });
    }
    
    group.finish();
}

fn tensor_access_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_access");
    
    let sizes = vec![100, 500, 1000];
    let mut rng = StdRng::seed_from_u64(42);
    
    for size in sizes {
        let shape = vec![size, size];
        let data = create_random_tensor(&shape, &mut rng);
        let tensor = Tensor::new(data, shape.clone()).unwrap();
        
        group.bench_with_input(BenchmarkId::new("sequential_access", size), &size, |b, _| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for i in 0..size {
                    for j in 0..size {
                        sum += tensor.get(&[i, j]).unwrap();
                    }
                }
                black_box(sum)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("random_access", size), &size, |b, _| {
            let mut access_rng = StdRng::seed_from_u64(123);
            b.iter(|| {
                let mut sum = 0.0f32;
                for _ in 0..1000 {
                    let i = access_rng.gen_range(0..size);
                    let j = access_rng.gen_range(0..size);
                    sum += tensor.get(&[i, j]).unwrap();
                }
                black_box(sum)
            })
        });
    }
    
    group.finish();
}

fn tensor_reshape_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_reshape");
    
    let mut rng = StdRng::seed_from_u64(42);
    let sizes = vec![(100, 100), (500, 500), (1000, 1000)];
    
    for (rows, cols) in sizes {
        let original_shape = vec![rows, cols];
        let data = create_random_tensor(&original_shape, &mut rng);
        let tensor = Tensor::new(data, original_shape.clone()).unwrap();
        
        let new_shape = vec![cols, rows]; // Transpose dimensions
        
        group.bench_with_input(
            BenchmarkId::new("reshape", format!("{}x{}", rows, cols)), 
            &(rows, cols), 
            |b, _| {
                b.iter(|| {
                    black_box(tensor.reshape(&new_shape).unwrap())
                })
            }
        );
        
        // Only benchmark transpose for 2D tensors
        if original_shape.len() == 2 {
            group.bench_with_input(
                BenchmarkId::new("transpose", format!("{}x{}", rows, cols)), 
                &(rows, cols), 
                |b, _| {
                    b.iter(|| {
                        black_box(tensor.transpose().unwrap())
                    })
                }
            );
        }
    }
    
    group.finish();
}

fn memory_intensive_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_intensive");
    group.sample_size(10); // Fewer samples for memory-intensive benchmarks
    
    let mut rng = StdRng::seed_from_u64(42);
    
    // Large tensor creation and operations
    let large_sizes = vec![2000, 3000];
    
    for size in large_sizes {
        group.bench_with_input(BenchmarkId::new("large_tensor_creation", size), &size, |b, &size| {
            b.iter(|| {
                let shape = vec![size, size];
                let data = create_random_tensor(&shape, &mut StdRng::seed_from_u64(42));
                black_box(Tensor::new(data, shape).unwrap())
            })
        });
        
        // Pre-create tensors for operations
        let shape = vec![size, size];
        let data1 = create_random_tensor(&shape, &mut rng);
        let data2 = create_random_tensor(&shape, &mut rng);
        let tensor1 = Tensor::new(data1, shape.clone()).unwrap();
        let tensor2 = Tensor::new(data2, shape.clone()).unwrap();
        
        group.bench_with_input(BenchmarkId::new("large_tensor_addition", size), &size, |b, _| {
            b.iter(|| {
                black_box((&tensor1 + &tensor2).unwrap())
            })
        });
    }
    
    group.finish();
}

fn tensor_workflow_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_workflows");
    
    let mut rng = StdRng::seed_from_u64(42);
    let size = 500;
    
    group.bench_function("complete_tensor_workflow", |b| {
        b.iter(|| {
            // Create initial tensors
            let tensor1 = Tensor::<f32>::randn(&mut StdRng::seed_from_u64(42), &[size, size]);
            let tensor2 = Tensor::<f32>::uniform(&mut StdRng::seed_from_u64(43), &[size, size], -1.0, 1.0);
            
            // Perform operations
            let sum = (&tensor1 + &tensor2).unwrap();
            let product = &sum * 2.0;
            let reshaped = product.reshape(&[size * 2, size / 2]).unwrap();
            
            black_box(reshaped)
        })
    });
    
    group.bench_function("tensor_chain_operations", |b| {
        let tensor = Tensor::<f32>::ones(&[size, size]);
        
        b.iter(|| {
            let mut result = tensor.clone();
            for i in 1..=10 {
                let scalar = i as f32;
                result = &result * scalar;
                result = (&result + &tensor).unwrap();
            }
            black_box(result)
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    tensor_creation_benchmarks,
    tensor_operations_benchmarks,
    tensor_access_benchmarks,
    tensor_reshape_benchmarks,
    memory_intensive_benchmarks,
    tensor_workflow_benchmarks
);

criterion_main!(benches);