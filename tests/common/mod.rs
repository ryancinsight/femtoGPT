use approx::AbsDiffEq;
use femto_gpt::graph::{CpuGraph, GraphError};
use femto_gpt::tensor::{Tensor, TensorId};
use proptest::prelude::*;
use rand::prelude::*;
use std::collections::HashMap;

/// Test utilities for tensor operations and assertions
pub mod tensor_utils {
    use super::*;

    /// Custom assertion macro for tensor equality with tolerance
    #[macro_export]
    macro_rules! assert_tensor_eq {
        ($left:expr, $right:expr, $tolerance:expr) => {{
            let left_data = $left.data();
            let right_data = $right.data();
            assert_eq!(
                left_data.len(),
                right_data.len(),
                "Tensor dimensions mismatch: {} vs {}",
                left_data.len(),
                right_data.len()
            );
            for (i, (l, r)) in left_data.iter().zip(right_data.iter()).enumerate() {
                assert!(
                    (l - r).abs() < $tolerance,
                    "Tensor values differ at index {}: {} vs {} (tolerance: {})",
                    i,
                    l,
                    r,
                    $tolerance
                );
            }
        }};
        ($left:expr, $right:expr) => {
            assert_tensor_eq!($left, $right, 1e-6);
        };
    }

    /// Custom assertion macro for tensor shape equality
    #[macro_export]
    macro_rules! assert_tensor_shape {
        ($tensor:expr, $expected_shape:expr) => {{
            let actual_shape = $tensor.shape();
            assert_eq!(
                actual_shape, $expected_shape,
                "Tensor shape mismatch: expected {:?}, got {:?}",
                $expected_shape, actual_shape
            );
        }};
    }

    /// Create a test tensor with specified shape and random values
    pub fn create_random_tensor(shape: &[usize], rng: &mut impl Rng) -> Vec<f32> {
        let size = shape.iter().product();
        (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    /// Create a test tensor with specified shape and sequential values
    pub fn create_sequential_tensor(shape: &[usize]) -> Vec<f32> {
        let size = shape.iter().product();
        (0..size).map(|i| i as f32).collect()
    }

    /// Create a test tensor with ones
    pub fn create_ones_tensor(shape: &[usize]) -> Vec<f32> {
        let size = shape.iter().product();
        vec![1.0; size]
    }

    /// Create a test tensor with zeros
    pub fn create_zeros_tensor(shape: &[usize]) -> Vec<f32> {
        let size = shape.iter().product();
        vec![0.0; size]
    }

    /// Generate property test strategies for tensor shapes
    pub fn tensor_shape_strategy() -> impl Strategy<Value = Vec<usize>> {
        prop::collection::vec(1..=10usize, 1..=4)
    }

    /// Generate property test strategies for small tensor shapes (for performance)
    pub fn small_tensor_shape_strategy() -> impl Strategy<Value = Vec<usize>> {
        prop::collection::vec(1..=5usize, 1..=3)
    }

    /// Generate property test strategies for tensor data
    pub fn tensor_data_strategy(size: usize) -> impl Strategy<Value = Vec<f32>> {
        prop::collection::vec(-10.0..10.0f32, size..=size)
    }
}

/// Test utilities for graph operations
pub mod graph_utils {
    use super::*;

    /// Create a test CPU graph
    pub fn create_test_cpu_graph() -> CpuGraph {
        CpuGraph::new()
    }

    /// Create a test GPU graph if available
    #[cfg(feature = "gpu")]
    pub fn create_test_gpu_graph() -> Result<femto_gpt::graph::gpu::GpuGraph, GraphError> {
        femto_gpt::graph::gpu::GpuGraph::new()
    }

    /// Test configuration for GPU/CPU variants
    pub struct TestConfig {
        pub use_gpu: bool,
        pub tolerance: f32,
        pub max_iterations: usize,
    }

    impl Default for TestConfig {
        fn default() -> Self {
            Self {
                use_gpu: false,
                tolerance: 1e-6,
                max_iterations: 1000,
            }
        }
    }

    /// Run a test with both CPU and GPU configurations (if available)
    pub fn run_dual_test<F>(test_fn: F) -> Result<(), GraphError>
    where
        F: Fn(TestConfig) -> Result<(), GraphError>,
    {
        // Always test CPU
        test_fn(TestConfig {
            use_gpu: false,
            ..Default::default()
        })?;

        // Test GPU if feature is enabled
        #[cfg(feature = "gpu")]
        {
            if create_test_gpu_graph().is_ok() {
                test_fn(TestConfig {
                    use_gpu: true,
                    tolerance: 1e-5, // Slightly more tolerance for GPU
                    ..Default::default()
                })?;
            }
        }

        Ok(())
    }
}

/// Test utilities for GPT model testing
pub mod model_utils {
    use super::*;
    use femto_gpt::gpt::GPT;
    use femto_gpt::tokenizer::{SimpleTokenizer, Tokenizer};

    /// Create a minimal test GPT model
    pub fn create_test_gpt<G>(
        graph: G,
        rng: &mut impl Rng,
    ) -> Result<GPT<G>, GraphError>
    where
        G: femto_gpt::graph::Graph,
    {
        let vocab_size = 10;
        let embedding_degree = 8;
        let num_tokens = 4;
        let num_layers = 2;
        let num_heads = 2;
        let head_size = embedding_degree / num_heads;
        let dropout = 0.0;

        GPT::new(
            rng,
            graph,
            None, // No pre-allocated batches for tests
            vocab_size,
            embedding_degree,
            num_tokens,
            num_layers,
            num_heads,
            head_size,
            dropout,
        )
    }

    /// Create a test tokenizer with simple vocabulary
    pub fn create_test_tokenizer() -> SimpleTokenizer {
        let test_text = "abcdefghij"; // 10 unique characters
        SimpleTokenizer::new(test_text)
    }

    /// Create test training data
    pub fn create_test_dataset() -> Vec<usize> {
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3] // Simple repeating pattern
    }
}

/// Test fixtures and data
pub mod fixtures {
    use super::*;

    /// Load or create test dataset for training tests
    pub fn get_shakespeare_sample() -> String {
        // Small sample of Shakespeare-like text for testing
        "To be or not to be, that is the question.\nWhether 'tis nobler in the mind to suffer\nThe slings and arrows of outrageous fortune,\nOr to take arms against a sea of troubles".to_string()
    }

    /// Get test model configurations
    pub fn get_test_model_configs() -> Vec<ModelConfig> {
        vec![
            ModelConfig {
                name: "tiny".to_string(),
                vocab_size: 10,
                embedding_degree: 8,
                num_tokens: 4,
                num_layers: 1,
                num_heads: 1,
            },
            ModelConfig {
                name: "small".to_string(),
                vocab_size: 20,
                embedding_degree: 16,
                num_tokens: 8,
                num_layers: 2,
                num_heads: 2,
            },
        ]
    }

    #[derive(Debug, Clone)]
    pub struct ModelConfig {
        pub name: String,
        pub vocab_size: usize,
        pub embedding_degree: usize,
        pub num_tokens: usize,
        pub num_layers: usize,
        pub num_heads: usize,
    }
}

/// Performance testing utilities
pub mod perf_utils {
    use std::time::{Duration, Instant};

    /// Measure execution time of a function
    pub fn measure_time<F, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }

    /// Performance assertion macro
    #[macro_export]
    macro_rules! assert_performance {
        ($expr:expr, $max_duration:expr, $description:expr) => {{
            let (result, duration) = $crate::common::perf_utils::measure_time(|| $expr);
            assert!(
                duration <= $max_duration,
                "{} took {:?}, expected <= {:?}",
                $description,
                duration,
                $max_duration
            );
            result
        }};
    }

    /// Memory usage tracking (simplified)
    pub struct MemoryTracker {
        start_usage: usize,
    }

    impl MemoryTracker {
        pub fn new() -> Self {
            Self { start_usage: 0 } // Simplified - would use actual memory tracking in production
        }

        pub fn current_usage(&self) -> usize {
            // Simplified implementation
            0
        }

        pub fn usage_since_start(&self) -> usize {
            self.current_usage().saturating_sub(self.start_usage)
        }
    }
}

/// Error handling utilities for tests
pub mod error_utils {
    use super::*;

    /// Assert that a function returns a specific error type
    #[macro_export]
    macro_rules! assert_error {
        ($expr:expr, $error_pattern:pat) => {{
            match $expr {
                Err($error_pattern) => (),
                Ok(val) => panic!("Expected error, got Ok({:?})", val),
                Err(e) => panic!("Expected error pattern, got different error: {:?}", e),
            }
        }};
    }

    /// Assert that a function returns any error
    #[macro_export]
    macro_rules! assert_any_error {
        ($expr:expr) => {{
            match $expr {
                Err(_) => (),
                Ok(val) => panic!("Expected error, got Ok({:?})", val),
            }
        }};
    }
}

// Re-export commonly used items
pub use tensor_utils::*;
pub use graph_utils::*;
pub use model_utils::*;
pub use fixtures::*;
pub use perf_utils::*;
pub use error_utils::*;