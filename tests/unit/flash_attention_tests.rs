use femto_gpt::funcs::{FlashAttention, Function, MatMul, Softmax, TrilMask, Coeff};
use femto_gpt::tensor::{GeneralTensor, Tensor, TensorOps};
use proptest::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::*;

    /// Test basic Flash Attention functionality
    #[test]
    fn test_flash_attention_basic_functionality() {
        let seq_len = 8;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Create test inputs with known values
        let q_data: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32 * 0.1).collect();
        let k_data: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let v_data: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 + 2.0) * 0.1).collect();

        let q = Tensor::new(q_data, vec![seq_len, head_dim]).unwrap();
        let k = Tensor::new(k_data, vec![seq_len, head_dim]).unwrap();
        let v = Tensor::new(v_data, vec![seq_len, head_dim]).unwrap();

        let mut flash_attn = FlashAttention::new(scale, false);
        let inputs = vec![
            &GeneralTensor::Float(q),
            &GeneralTensor::Float(k),
            &GeneralTensor::Float(v),
        ];

        let result = flash_attn.run(&inputs, false);
        assert!(result.is_ok(), "Flash attention should execute successfully");
        
        let output = result.unwrap();
        assert_tensor_shape!(output, vec![seq_len, head_dim]);
        
        // Verify output is not all zeros (basic sanity check)
        let sum: f32 = output.data().iter().sum();
        assert!(sum.abs() > 1e-6, "Output should not be all zeros");
    }

    /// Test Flash Attention with causal masking
    #[test]
    fn test_flash_attention_causal_masking() {
        let seq_len = 4;
        let head_dim = 2;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Create simple test data
        let q_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let k_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let v_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let q = Tensor::new(q_data, vec![seq_len, head_dim]).unwrap();
        let k = Tensor::new(k_data, vec![seq_len, head_dim]).unwrap();
        let v = Tensor::new(v_data, vec![seq_len, head_dim]).unwrap();

        let mut flash_attn = FlashAttention::new(scale, true);
        let inputs = vec![
            &GeneralTensor::Float(q),
            &GeneralTensor::Float(k),
            &GeneralTensor::Float(v),
        ];

        let result = flash_attn.run(&inputs, false);
        assert!(result.is_ok(), "Causal Flash attention should execute successfully");
        
        let output = result.unwrap();
        assert_tensor_shape!(output, vec![seq_len, head_dim]);
    }

    /// Test numerical equivalence with standard attention (simplified)
    #[test]
    fn test_numerical_equivalence_simple() {
        let seq_len = 4;
        let head_dim = 2;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Simple test case for numerical verification
        let q_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let k_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let v_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let q = Tensor::new(q_data.clone(), vec![seq_len, head_dim]).unwrap();
        let k = Tensor::new(k_data.clone(), vec![seq_len, head_dim]).unwrap();
        let v = Tensor::new(v_data.clone(), vec![seq_len, head_dim]).unwrap();

        // Flash Attention output
        let mut flash_attn = FlashAttention::new(scale, false);
        let flash_inputs = vec![
            &GeneralTensor::Float(q.clone()),
            &GeneralTensor::Float(k.clone()),
            &GeneralTensor::Float(v.clone()),
        ];
        let flash_output = flash_attn.run(&flash_inputs, false).unwrap();

        // Standard attention computation (simplified)
        let standard_output = compute_standard_attention(&q, &k, &v, scale, false).unwrap();

        // Compare outputs with tolerance
        assert_tensor_eq!(flash_output, standard_output, 1e-4);
    }

    /// Test Flash Attention with different block sizes
    #[test]
    fn test_different_block_sizes() {
        let seq_len = 16;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = create_test_tensor(&[seq_len, head_dim], 0.1);
        let k = create_test_tensor(&[seq_len, head_dim], 0.2);
        let v = create_test_tensor(&[seq_len, head_dim], 0.3);

        let inputs = vec![
            &GeneralTensor::Float(q),
            &GeneralTensor::Float(k),
            &GeneralTensor::Float(v),
        ];

        // Test with different block sizes
        let block_sizes = vec![(4, 4), (8, 8), (16, 16)];
        let mut outputs = Vec::new();

        for (block_q, block_k) in block_sizes {
            let mut flash_attn = FlashAttention::with_block_sizes(scale, false, block_q, block_k);
            let output = flash_attn.run(&inputs, false).unwrap();
            outputs.push(output);
        }

        // All outputs should be numerically equivalent
        for i in 1..outputs.len() {
            assert_tensor_eq!(outputs[0], outputs[i], 1e-5);
        }
    }

    /// Test Flash Attention error handling
    #[test]
    fn test_error_handling() {
        let scale = 0.5;
        let mut flash_attn = FlashAttention::new(scale, false);

        // Test with wrong number of inputs
        let q = create_test_tensor(&[4, 2], 0.1);
        let wrong_inputs = vec![&GeneralTensor::Float(q)];
        
        let result = flash_attn.run(&wrong_inputs, false);
        assert!(result.is_err(), "Should fail with wrong number of inputs");

        // Test with mismatched dimensions
        let q = create_test_tensor(&[4, 2], 0.1);
        let k = create_test_tensor(&[4, 3], 0.2); // Different head dimension
        let v = create_test_tensor(&[4, 2], 0.3);

        let mismatched_inputs = vec![
            &GeneralTensor::Float(q),
            &GeneralTensor::Float(k),
            &GeneralTensor::Float(v),
        ];

        let result = flash_attn.run(&mismatched_inputs, false);
        assert!(result.is_err(), "Should fail with mismatched dimensions");
    }

    /// Property-based test for Flash Attention
    proptest! {
        #[test]
        fn test_flash_attention_properties(
            seq_len in 2usize..32,
            head_dim in 2usize..16,
            seed in 0u64..1000
        ) {
            let scale = 1.0 / (head_dim as f32).sqrt();
            
            let q = create_seeded_tensor(&[seq_len, head_dim], seed);
            let k = create_seeded_tensor(&[seq_len, head_dim], seed + 1);
            let v = create_seeded_tensor(&[seq_len, head_dim], seed + 2);

            let mut flash_attn = FlashAttention::new(scale, false);
            let inputs = vec![
                &GeneralTensor::Float(q),
                &GeneralTensor::Float(k),
                &GeneralTensor::Float(v),
            ];

            let result = flash_attn.run(&inputs, false);
            prop_assert!(result.is_ok(), "Flash attention should always succeed with valid inputs");
            
            let output = result.unwrap();
            prop_assert_eq!(output.shape(), &[seq_len, head_dim]);
            
            // Check that output is finite
            for &val in output.data() {
                prop_assert!(val.is_finite(), "All output values should be finite");
            }
        }
    }

    /// Performance test comparing Flash Attention with standard attention
    #[test]
    fn test_performance_comparison() {
        let seq_lens = vec![64, 128, 256];
        let head_dim = 64;
        let scale = 1.0 / (head_dim as f32).sqrt();

        for seq_len in seq_lens {
            let q = create_test_tensor(&[seq_len, head_dim], 0.1);
            let k = create_test_tensor(&[seq_len, head_dim], 0.2);
            let v = create_test_tensor(&[seq_len, head_dim], 0.3);

            let inputs = vec![
                &GeneralTensor::Float(q.clone()),
                &GeneralTensor::Float(k.clone()),
                &GeneralTensor::Float(v.clone()),
            ];

            // Measure Flash Attention time
            let flash_time = measure_time(|| {
                let mut flash_attn = FlashAttention::new(scale, false);
                flash_attn.run(&inputs, false).unwrap()
            });

            // Measure standard attention time
            let standard_time = measure_time(|| {
                compute_standard_attention(&q, &k, &v, scale, false).unwrap()
            });

            println!(
                "Seq len {}: Flash Attention: {:.2}ms, Standard: {:.2}ms",
                seq_len, flash_time, standard_time
            );

            // For small sequences, Flash Attention might be slower due to overhead
            // For larger sequences, it should be competitive or faster
            if seq_len >= 256 {
                assert!(
                    flash_time <= standard_time * 2.0,
                    "Flash Attention should be competitive for larger sequences"
                );
            }
        }
    }

    /// Memory usage test (basic validation)
    #[test]
    fn test_memory_efficiency() {
        let seq_len = 128;
        let head_dim = 64;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = create_test_tensor(&[seq_len, head_dim], 0.1);
        let k = create_test_tensor(&[seq_len, head_dim], 0.2);
        let v = create_test_tensor(&[seq_len, head_dim], 0.3);

        let inputs = vec![
            &GeneralTensor::Float(q),
            &GeneralTensor::Float(k),
            &GeneralTensor::Float(v),
        ];

        // Test with different block sizes to verify memory usage
        let small_block = FlashAttention::with_block_sizes(scale, false, 16, 16);
        let large_block = FlashAttention::with_block_sizes(scale, false, 64, 64);

        // Both should produce the same result
        let mut small_attn = small_block;
        let mut large_attn = large_block;

        let small_output = small_attn.run(&inputs, false).unwrap();
        let large_output = large_attn.run(&inputs, false).unwrap();

        assert_tensor_eq!(small_output, large_output, 1e-5);
    }

    /// Test gradient computation (basic structure)
    #[test]
    fn test_gradient_computation() {
        let seq_len = 4;
        let head_dim = 2;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = create_test_tensor(&[seq_len, head_dim], 0.1);
        let k = create_test_tensor(&[seq_len, head_dim], 0.2);
        let v = create_test_tensor(&[seq_len, head_dim], 0.3);

        let inputs = vec![
            &GeneralTensor::Float(q.clone()),
            &GeneralTensor::Float(k.clone()),
            &GeneralTensor::Float(v.clone()),
        ];

        let flash_attn = FlashAttention::new(scale, false);
        let grad_output = create_test_tensor(&[seq_len, head_dim], 1.0);

        let gradients = flash_attn.grad(&inputs, &grad_output);
        assert!(gradients.is_ok(), "Gradient computation should succeed");

        let grads = gradients.unwrap();
        assert_eq!(grads.len(), 3, "Should return gradients for Q, K, V");
        
        // Verify gradient shapes
        assert_tensor_shape!(grads[0], q.shape().to_vec());
        assert_tensor_shape!(grads[1], k.shape().to_vec());
        assert_tensor_shape!(grads[2], v.shape().to_vec());
    }
}

/// Helper function to compute standard attention for comparison
fn compute_standard_attention(
    q: &Tensor<f32>,
    k: &Tensor<f32>,
    v: &Tensor<f32>,
    scale: f32,
    causal: bool,
) -> Result<Tensor<f32>, femto_gpt::tensor::TensorError> {
    // Q @ K^T
    let k_t = k.transpose()?;
    let scores = (q.matmul(&k_t)? * scale)?;
    
    // Apply causal mask if needed
    let masked_scores = if causal {
        let seq_len = q.shape()[0];
        let mut masked = scores.clone();
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                masked.set(&[i, j], f32::NEG_INFINITY)?;
            }
        }
        masked
    } else {
        scores
    };
    
    // Softmax
    let mut softmax = Softmax::new();
    let attention_weights = softmax.run(
        &[&GeneralTensor::Float(masked_scores)],
        false
    )?;
    
    // Attention @ V
    attention_weights.matmul(v)
}

/// Helper function to create a test tensor with seeded random values
fn create_seeded_tensor(shape: &[usize], seed: u64) -> Tensor<f32> {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    
    let mut rng = StdRng::seed_from_u64(seed);
    let size = shape.iter().product();
    let data: Vec<f32> = (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect();
    
    Tensor::new(data, shape.to_vec()).unwrap()
}