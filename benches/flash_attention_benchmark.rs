use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use femto_gpt::funcs::{FlashAttention, Function, Softmax, Coeff, MatMul};
use femto_gpt::tensor::{GeneralTensor, Tensor, TensorOps};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Create a test tensor with random values
fn create_random_tensor(shape: &[usize], seed: u64) -> Tensor<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let size = shape.iter().product();
    let data: Vec<f32> = (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect();
    Tensor::raw(shape, data).unwrap()
}

/// Compute standard attention for comparison
fn compute_standard_attention(
    q: &Tensor<f32>,
    k: &Tensor<f32>,
    v: &Tensor<f32>,
    scale: f32,
) -> Result<Tensor<f32>, femto_gpt::tensor::TensorError> {
    // Q @ K^T
    let k_t = k.transpose()?;
    let mut matmul = MatMul::new();
    let qk = matmul.run(&[&GeneralTensor::Float(q.clone()), &GeneralTensor::Float(k_t)], false)?;
    let mut coeff = Coeff::new(scale);
    let scores = coeff.run(&[&GeneralTensor::Float(qk)], false)?;
    
    // Softmax
    let mut softmax = Softmax::new();
    let attention_weights = softmax.run(
        &[&GeneralTensor::Float(scores)],
        false
    )?;
    
    // Attention @ V
    let mut matmul2 = MatMul::new();
    matmul2.run(&[&GeneralTensor::Float(attention_weights), &GeneralTensor::Float(v.clone())], false)
}

/// Benchmark Flash Attention vs Standard Attention across sequence lengths
fn attention_comparison_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_comparison");
    
    // Test different sequence lengths with fixed head dimension
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let seq_lengths = vec![64, 128, 256];
    
    for seq_len in seq_lengths {
        let q = create_random_tensor(&[seq_len, head_dim], 42);
        let k = create_random_tensor(&[seq_len, head_dim], 43);
        let v = create_random_tensor(&[seq_len, head_dim], 44);
        
        // Benchmark Flash Attention
        group.bench_with_input(
            BenchmarkId::new("flash_attention", seq_len),
            &seq_len,
            |b, _| {
                let q_gen = GeneralTensor::Float(q.clone());
                let k_gen = GeneralTensor::Float(k.clone());
                let v_gen = GeneralTensor::Float(v.clone());
                let inputs = vec![&q_gen, &k_gen, &v_gen];
                
                b.iter(|| {
                    let mut flash_attn = FlashAttention::new(scale, false);
                    black_box(flash_attn.run(&inputs, false).unwrap())
                })
            },
        );
        
        // Benchmark Standard Attention
        group.bench_with_input(
            BenchmarkId::new("standard_attention", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    black_box(compute_standard_attention(&q, &k, &v, scale).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark Flash Attention with different block sizes
fn block_size_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash_attention_block_sizes");
    
    let seq_len = 256;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    let q = create_random_tensor(&[seq_len, head_dim], 42);
    let k = create_random_tensor(&[seq_len, head_dim], 43);
    let v = create_random_tensor(&[seq_len, head_dim], 44);
    
    let q_gen = GeneralTensor::Float(q);
    let k_gen = GeneralTensor::Float(k);
    let v_gen = GeneralTensor::Float(v);
    let inputs = vec![&q_gen, &k_gen, &v_gen];
    
    let block_sizes = vec![16, 32, 64, 128];
    
    for block_size in block_sizes {
        group.bench_with_input(
            BenchmarkId::new("block_size", block_size),
            &block_size,
            |b, &block_size| {
                b.iter(|| {
                    let mut flash_attn = FlashAttention::with_block_sizes(
                        scale, false, block_size, block_size
                    );
                    black_box(flash_attn.run(&inputs, false).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark Flash Attention with causal masking
fn causal_attention_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("causal_attention");
    
    let seq_lengths = vec![64, 128, 256];
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    for seq_len in seq_lengths {
        let q = create_random_tensor(&[seq_len, head_dim], 42);
        let k = create_random_tensor(&[seq_len, head_dim], 43);
        let v = create_random_tensor(&[seq_len, head_dim], 44);
        
        let q_gen = GeneralTensor::Float(q);
        let k_gen = GeneralTensor::Float(k);
        let v_gen = GeneralTensor::Float(v);
        let inputs = vec![&q_gen, &k_gen, &v_gen];
        
        // Non-causal Flash Attention
        group.bench_with_input(
            BenchmarkId::new("non_causal", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    let mut flash_attn = FlashAttention::new(scale, false);
                    black_box(flash_attn.run(&inputs, false).unwrap())
                })
            },
        );
        
        // Causal Flash Attention
        group.bench_with_input(
            BenchmarkId::new("causal", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    let mut flash_attn = FlashAttention::new(scale, true);
                    black_box(flash_attn.run(&inputs, false).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory usage patterns
fn memory_usage_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    group.sample_size(10); // Fewer samples for memory-intensive tests
    
    let seq_lengths = vec![128, 256];
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    for seq_len in seq_lengths {
        let q = create_random_tensor(&[seq_len, head_dim], 42);
        let k = create_random_tensor(&[seq_len, head_dim], 43);
        let v = create_random_tensor(&[seq_len, head_dim], 44);
        
        let q_gen = GeneralTensor::Float(q.clone());
        let k_gen = GeneralTensor::Float(k.clone());
        let v_gen = GeneralTensor::Float(v.clone());
        let inputs = vec![&q_gen, &k_gen, &v_gen];
        
        // Small block size (memory efficient)
        group.bench_with_input(
            BenchmarkId::new("small_blocks", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    let mut flash_attn = FlashAttention::with_block_sizes(scale, false, 16, 16);
                    black_box(flash_attn.run(&inputs, false).unwrap())
                })
            },
        );
        
        // Large block size (potentially faster but more memory)
        group.bench_with_input(
            BenchmarkId::new("large_blocks", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    let mut flash_attn = FlashAttention::with_block_sizes(scale, false, 64, 64);
                    black_box(flash_attn.run(&inputs, false).unwrap())
                })
            },
        );
        
        // Standard attention (baseline memory usage)
        group.bench_with_input(
            BenchmarkId::new("standard_baseline", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    black_box(compute_standard_attention(&q, &k, &v, scale).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark different head dimensions
fn head_dimension_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("head_dimensions");
    
    let seq_len = 128;
    let head_dims = vec![32, 64, 128];
    
    for head_dim in head_dims {
        let scale = 1.0 / (head_dim as f32).sqrt();
        
        let q = create_random_tensor(&[seq_len, head_dim], 42);
        let k = create_random_tensor(&[seq_len, head_dim], 43);
        let v = create_random_tensor(&[seq_len, head_dim], 44);
        
        let q_gen = GeneralTensor::Float(q.clone());
        let k_gen = GeneralTensor::Float(k.clone());
        let v_gen = GeneralTensor::Float(v.clone());
        let inputs = vec![&q_gen, &k_gen, &v_gen];
        
        // Flash Attention
        group.bench_with_input(
            BenchmarkId::new("flash_attention", head_dim),
            &head_dim,
            |b, _| {
                b.iter(|| {
                    let mut flash_attn = FlashAttention::new(scale, false);
                    black_box(flash_attn.run(&inputs, false).unwrap())
                })
            },
        );
        
        // Standard Attention
        group.bench_with_input(
            BenchmarkId::new("standard_attention", head_dim),
            &head_dim,
            |b, _| {
                b.iter(|| {
                    black_box(compute_standard_attention(&q, &k, &v, scale).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark gradient computation
fn gradient_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_computation");
    
    let seq_len = 128;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    let q = create_random_tensor(&[seq_len, head_dim], 42);
    let k = create_random_tensor(&[seq_len, head_dim], 43);
    let v = create_random_tensor(&[seq_len, head_dim], 44);
    let grad_output = create_random_tensor(&[seq_len, head_dim], 45);
    
    let q_gen = GeneralTensor::Float(q);
    let k_gen = GeneralTensor::Float(k);
    let v_gen = GeneralTensor::Float(v);
    let inputs = vec![&q_gen, &k_gen, &v_gen];
    
    group.bench_function("flash_attention_grad", |b| {
        b.iter(|| {
            let flash_attn = FlashAttention::new(scale, false);
            black_box(flash_attn.grad(&inputs, &grad_output).unwrap())
        })
    });
    
    group.finish();
}

/// End-to-end workflow benchmark
fn workflow_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_workflow");
    
    let seq_len = 256;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    let q = create_random_tensor(&[seq_len, head_dim], 42);
    let k = create_random_tensor(&[seq_len, head_dim], 43);
    let v = create_random_tensor(&[seq_len, head_dim], 44);
    let grad_output = create_random_tensor(&[seq_len, head_dim], 45);
    
    let q_gen = GeneralTensor::Float(q.clone());
    let k_gen = GeneralTensor::Float(k.clone());
    let v_gen = GeneralTensor::Float(v.clone());
    let inputs = vec![&q_gen, &k_gen, &v_gen];
    
    // Complete forward + backward pass for Flash Attention
    group.bench_function("flash_attention_forward_backward", |b| {
        b.iter(|| {
            let mut flash_attn = FlashAttention::new(scale, false);
            let output = flash_attn.run(&inputs, true).unwrap();
            let gradients = flash_attn.grad(&inputs, &grad_output).unwrap();
            black_box((output, gradients))
        })
    });
    
    // Complete forward + backward pass for Standard Attention
    group.bench_function("standard_attention_forward_backward", |b| {
        b.iter(|| {
            let output = compute_standard_attention(&q, &k, &v, scale).unwrap();
            // Simplified gradient computation for comparison
            let grad_q = Tensor::<f32>::zeros(q.shape());
            let grad_k = Tensor::<f32>::zeros(k.shape());
            let grad_v = Tensor::<f32>::zeros(v.shape());
            black_box((output, vec![grad_q, grad_k, grad_v]))
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    attention_comparison_benchmarks,
    block_size_benchmarks,
    causal_attention_benchmarks,
    memory_usage_benchmarks,
    head_dimension_benchmarks,
    gradient_benchmarks,
    workflow_benchmarks
);
criterion_main!(benches);