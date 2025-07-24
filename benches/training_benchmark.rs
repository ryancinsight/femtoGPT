use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use femto_gpt::gpt::GPT;
use femto_gpt::graph::CpuGraph;
use femto_gpt::optimizer::AdamW;
use femto_gpt::tokenizer::{SimpleTokenizer, Tokenizer};
use rand::prelude::*;

fn create_test_gpt(vocab_size: usize, embedding_degree: usize, num_layers: usize) -> GPT<CpuGraph> {
    let mut rng = StdRng::seed_from_u64(42);
    let graph = CpuGraph::new();
    let num_tokens = 32;
    let num_heads = 4;
    let head_size = embedding_degree / num_heads;
    let dropout = 0.0;

    GPT::new(
        &mut rng,
        graph,
        None,
        vocab_size,
        embedding_degree,
        num_tokens,
        num_layers,
        num_heads,
        head_size,
        dropout,
    ).unwrap()
}

fn create_test_dataset(size: usize, vocab_size: usize) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(123);
    (0..size).map(|_| rng.gen_range(0..vocab_size)).collect()
}

fn model_initialization_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_initialization");
    
    let configs = vec![
        ("tiny", 50, 64, 2),
        ("small", 100, 128, 4),
        ("medium", 200, 256, 6),
    ];
    
    for (name, vocab_size, embedding_degree, num_layers) in configs {
        group.bench_with_input(
            BenchmarkId::new("gpt_initialization", name),
            &(vocab_size, embedding_degree, num_layers),
            |b, &(vocab_size, embedding_degree, num_layers)| {
                b.iter(|| {
                    black_box(create_test_gpt(vocab_size, embedding_degree, num_layers))
                })
            }
        );
    }
    
    group.finish();
}

fn inference_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference");
    
    let configs = vec![
        ("tiny", 50, 64, 2),
        ("small", 100, 128, 4),
    ];
    
    for (name, vocab_size, embedding_degree, num_layers) in configs {
        let mut gpt = create_test_gpt(vocab_size, embedding_degree, num_layers);
        gpt.sync().unwrap();
        
        let tokenizer = SimpleTokenizer::new(&(0..vocab_size).map(|i| (i as u8 as char)).collect::<String>());
        let prompt_tokens = vec![0, 1, 2, 3]; // Simple prompt
        
        group.bench_with_input(
            BenchmarkId::new("text_generation_64_tokens", name),
            &name,
            |b, _| {
                b.iter(|| {
                    let mut rng = StdRng::seed_from_u64(42);
                    black_box(gpt.infer(
                        &mut rng,
                        &prompt_tokens,
                        64, // Generate 64 tokens
                        0.5, // Temperature
                        |_| {}, // No callback
                    ).unwrap())
                })
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("text_generation_16_tokens", name),
            &name,
            |b, _| {
                b.iter(|| {
                    let mut rng = StdRng::seed_from_u64(42);
                    black_box(gpt.infer(
                        &mut rng,
                        &prompt_tokens,
                        16, // Generate 16 tokens
                        0.5, // Temperature
                        |_| {}, // No callback
                    ).unwrap())
                })
            }
        );
    }
    
    group.finish();
}

fn training_step_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_steps");
    group.sample_size(10); // Fewer samples for training benchmarks
    
    let configs = vec![
        ("tiny", 50, 64, 2),
        ("small", 100, 128, 4),
    ];
    
    for (name, vocab_size, embedding_degree, num_layers) in configs {
        let mut gpt = create_test_gpt(vocab_size, embedding_degree, num_layers);
        gpt.sync().unwrap();
        
        let dataset = create_test_dataset(1000, vocab_size);
        let batch_size = 8;
        let optimizer = AdamW::new();
        let learning_rate = |_step| 0.001f32;
        
        group.bench_with_input(
            BenchmarkId::new("single_training_step", name),
            &name,
            |b, _| {
                b.iter(|| {
                    // Simulate a single training step
                    let result = gpt.train_cpu(
                        &dataset,
                        1, // Just one step
                        batch_size,
                        None,
                        &optimizer,
                        learning_rate,
                        |_| Ok(()),
                    );
                    black_box(result)
                })
            }
        );
    }
    
    group.finish();
}

fn batch_processing_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");
    
    let vocab_size = 100;
    let embedding_degree = 128;
    let num_layers = 4;
    let mut gpt = create_test_gpt(vocab_size, embedding_degree, num_layers);
    gpt.sync().unwrap();
    
    let batch_sizes = vec![1, 4, 8, 16, 32];
    
    for batch_size in batch_sizes {
        let dataset = create_test_dataset(batch_size * 64, vocab_size); // Enough data for batch
        
        group.bench_with_input(
            BenchmarkId::new("batch_inference", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let mut rng = StdRng::seed_from_u64(42);
                    // Simulate batch inference by running multiple single inferences
                    let mut results = Vec::new();
                    for i in 0..batch_size {
                        let start_idx = i * 4;
                        let prompt = &dataset[start_idx..start_idx + 4];
                        let result = gpt.infer(
                            &mut rng,
                            prompt,
                            16,
                            0.5,
                            |_| {},
                        ).unwrap();
                        results.push(result);
                    }
                    black_box(results)
                })
            }
        );
    }
    
    group.finish();
}

fn memory_usage_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    group.sample_size(10);
    
    let configs = vec![
        ("1M_params", 100, 128, 2),
        ("5M_params", 200, 256, 4),
    ];
    
    for (name, vocab_size, embedding_degree, num_layers) in configs {
        group.bench_with_input(
            BenchmarkId::new("model_memory_footprint", name),
            &(vocab_size, embedding_degree, num_layers),
            |b, &(vocab_size, embedding_degree, num_layers)| {
                b.iter(|| {
                    let mut gpt = create_test_gpt(vocab_size, embedding_degree, num_layers);
                    gpt.sync().unwrap();
                    
                    // Perform some operations to allocate memory
                    let dataset = create_test_dataset(100, vocab_size);
                    let mut rng = StdRng::seed_from_u64(42);
                    let _result = gpt.infer(&mut rng, &dataset[0..4], 10, 0.5, |_| {});
                    
                    black_box(gpt)
                })
            }
        );
    }
    
    group.finish();
}

fn tokenizer_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenizer");
    
    let vocab_sizes = vec![100, 500, 1000];
    let text_lengths = vec![100, 1000, 10000];
    
    for vocab_size in vocab_sizes {
        let vocab_text: String = (0..vocab_size).map(|i| (i as u8 as char)).collect();
        let tokenizer = SimpleTokenizer::new(&vocab_text);
        
        for text_length in &text_lengths {
            let test_text: String = (0..*text_length)
                .map(|i| vocab_text.chars().nth(i % vocab_size).unwrap())
                .collect();
            
            group.bench_with_input(
                BenchmarkId::new("tokenize", format!("vocab{}_text{}", vocab_size, text_length)),
                &test_text,
                |b, text| {
                    b.iter(|| {
                        black_box(tokenizer.tokenize(text))
                    })
                }
            );
            
            let tokens = tokenizer.tokenize(&test_text);
            group.bench_with_input(
                BenchmarkId::new("untokenize", format!("vocab{}_text{}", vocab_size, text_length)),
                &tokens,
                |b, tokens| {
                    b.iter(|| {
                        black_box(tokenizer.untokenize(tokens))
                    })
                }
            );
        }
    }
    
    group.finish();
}

fn end_to_end_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end");
    group.sample_size(5); // Very few samples for end-to-end benchmarks
    
    group.bench_function("complete_training_workflow", |b| {
        b.iter(|| {
            // Create model
            let vocab_size = 50;
            let mut gpt = create_test_gpt(vocab_size, 64, 2);
            gpt.sync().unwrap();
            
            // Create dataset
            let dataset = create_test_dataset(200, vocab_size);
            
            // Train for a few steps
            let result = gpt.train_cpu(
                &dataset,
                5, // 5 training steps
                4, // batch size
                None,
                &AdamW::new(),
                |_| 0.001f32,
                |_| Ok(()),
            );
            
            black_box(result)
        })
    });
    
    group.bench_function("complete_inference_workflow", |b| {
        b.iter(|| {
            // Create and initialize model
            let vocab_size = 50;
            let mut gpt = create_test_gpt(vocab_size, 64, 2);
            gpt.sync().unwrap();
            
            // Create tokenizer
            let vocab_text: String = (0..vocab_size).map(|i| (i as u8 as char)).collect();
            let tokenizer = SimpleTokenizer::new(&vocab_text);
            
            // Generate text
            let prompt = "hello";
            let prompt_tokens = tokenizer.tokenize(prompt);
            let mut rng = StdRng::seed_from_u64(42);
            
            let generated_tokens = gpt.infer(
                &mut rng,
                &prompt_tokens,
                50,
                0.7,
                |_| {},
            ).unwrap();
            
            let generated_text = tokenizer.untokenize(&generated_tokens);
            black_box(generated_text)
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    model_initialization_benchmarks,
    inference_benchmarks,
    training_step_benchmarks,
    batch_processing_benchmarks,
    memory_usage_benchmarks,
    tokenizer_benchmarks,
    end_to_end_benchmarks
);

criterion_main!(benches);