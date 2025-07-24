use femto_gpt::funcs::*;
use femto_gpt::tensor::{GeneralTensor, Tensor, TensorError};
use proptest::prelude::*;
use rand::prelude::*;

// Import test utilities
#[path = "../common/mod.rs"]
mod common;
use common::*;

/// Test utilities specific to function testing
mod function_test_utils {
    use super::*;

    /// Numerical gradient checking for functions
    pub fn numerical_gradient_check<F>(
        function: &mut F,
        inputs: &[&GeneralTensor],
        epsilon: f32,
        tolerance: f32,
    ) -> Result<(), TensorError>
    where
        F: Function,
    {
        // Forward pass to get output
        let output = function.run(inputs, true)?;
        let out_grad = Tensor::ones(output.shape());
        
        // Get analytical gradients
        let analytical_grads = function.grad(inputs, &out_grad)?;
        
        // Compute numerical gradients for each input
        for (input_idx, input) in inputs.iter().enumerate() {
            let input_tensor = input.as_float()?;
            let mut numerical_grad = Tensor::zeros(input_tensor.shape());
            
            for i in 0..input_tensor.size() {
                // Create perturbed inputs (forward)
                let mut input_data_forward = input_tensor.data().clone();
                input_data_forward[i] += epsilon;
                let input_forward = Tensor::new(input_data_forward, input_tensor.shape().to_vec())?;
                let general_forward = GeneralTensor::Float(input_forward);
                
                // Create perturbed inputs (backward)
                let mut input_data_backward = input_tensor.data().clone();
                input_data_backward[i] -= epsilon;
                let input_backward = Tensor::new(input_data_backward, input_tensor.shape().to_vec())?;
                let general_backward = GeneralTensor::Float(input_backward);
                
                // Create modified input arrays
                let mut inputs_forward = inputs.to_vec();
                let mut inputs_backward = inputs.to_vec();
                inputs_forward[input_idx] = &general_forward;
                inputs_backward[input_idx] = &general_backward;
                
                // Compute outputs
                let output_forward = function.run(&inputs_forward.iter().collect::<Vec<_>>(), true)?;
                let output_backward = function.run(&inputs_backward.iter().collect::<Vec<_>>(), true)?;
                
                // Compute numerical gradient
                let grad_sum_forward: f32 = output_forward.data().iter().sum();
                let grad_sum_backward: f32 = output_backward.data().iter().sum();
                let numerical_grad_val = (grad_sum_forward - grad_sum_backward) / (2.0 * epsilon);
                
                // Store in numerical gradient tensor
                let mut grad_data = numerical_grad.data_mut();
                grad_data[i] = numerical_grad_val;
            }
            
            // Compare analytical and numerical gradients
            let analytical_grad = &analytical_grads[input_idx];
            for (i, (&analytical, &numerical)) in analytical_grad.data().iter()
                .zip(numerical_grad.data().iter()).enumerate() {
                let diff = (analytical - numerical).abs();
                if diff > tolerance {
                    eprintln!("Gradient mismatch at input {} index {}: analytical={}, numerical={}, diff={}",
                        input_idx, i, analytical, numerical, diff);
                    return Err(TensorError::UnexpectedShape);
                }
            }
        }
        
        Ok(())
    }
}

/// Tests for GELU activation function
#[cfg(test)]
mod gelu_tests {
    use super::*;

    #[test]
    fn test_gelu_forward() {
        let input_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let input = Tensor::new(input_data, vec![5]).unwrap();
        let general_input = GeneralTensor::Float(input);
        
        let mut gelu = Gelu::new();
        let output = gelu.run(&[&general_input], false).unwrap();
        
        // GELU should be approximately 0 at x=0, and approach identity for large positive x
        assert_eq!(output.shape(), &[5]);
        assert!(output.data()[2].abs() < 1e-6); // GELU(0) ≈ 0
        assert!(output.data()[4] > 1.5); // GELU(2) should be close to 2
    }

    #[test]
    fn test_gelu_gradient() {
        let input_data = vec![-1.0, 0.0, 1.0];
        let input = Tensor::new(input_data, vec![3]).unwrap();
        let general_input = GeneralTensor::Float(input);
        
        let mut gelu = Gelu::new();
        
        // Test gradient computation
        function_test_utils::numerical_gradient_check(
            &mut gelu,
            &[&general_input],
            1e-5,
            1e-3,
        ).unwrap();
    }

    proptest! {
        #[test]
        fn test_gelu_properties(
            input_data in prop::collection::vec(-5.0..5.0f32, 1..20),
        ) {
            let input = Tensor::new(input_data.clone(), vec![input_data.len()]).unwrap();
            let general_input = GeneralTensor::Float(input);
            
            let mut gelu = Gelu::new();
            let output = gelu.run(&[&general_input], false).unwrap();
            
            // GELU should preserve shape
            assert_eq!(output.shape(), &[input_data.len()]);
            
            // GELU should be finite for all finite inputs
            for &val in output.data() {
                assert!(val.is_finite());
            }
        }
    }
}

/// Tests for ReLU activation function
#[cfg(test)]
mod relu_tests {
    use super::*;

    #[test]
    fn test_relu_forward() {
        let input_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let input = Tensor::new(input_data, vec![5]).unwrap();
        let general_input = GeneralTensor::Float(input);
        
        let mut relu = ReLU::new();
        let output = relu.run(&[&general_input], false).unwrap();
        
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];
        assert_eq!(output.data(), &expected);
    }

    #[test]
    fn test_relu_gradient() {
        let input_data = vec![-1.0, 0.0, 1.0];
        let input = Tensor::new(input_data, vec![3]).unwrap();
        let general_input = GeneralTensor::Float(input);
        
        let mut relu = ReLU::new();
        let output = relu.run(&[&general_input], true).unwrap();
        let out_grad = Tensor::ones(output.shape());
        
        let grads = relu.grad(&[&general_input], &out_grad).unwrap();
        let expected_grad = vec![0.0, 0.0, 1.0]; // Gradient is 1 for positive, 0 for negative
        
        assert_eq!(grads[0].data(), &expected_grad);
    }

    proptest! {
        #[test]
        fn test_relu_properties(
            input_data in prop::collection::vec(-10.0..10.0f32, 1..50),
        ) {
            let input = Tensor::new(input_data.clone(), vec![input_data.len()]).unwrap();
            let general_input = GeneralTensor::Float(input);
            
            let mut relu = ReLU::new();
            let output = relu.run(&[&general_input], false).unwrap();
            
            // ReLU should clamp negative values to 0 and preserve positive values
            for (i, (&input_val, &output_val)) in input_data.iter().zip(output.data().iter()).enumerate() {
                if input_val >= 0.0 {
                    assert_eq!(output_val, input_val, "ReLU should preserve positive values at index {}", i);
                } else {
                    assert_eq!(output_val, 0.0, "ReLU should clamp negative values to 0 at index {}", i);
                }
            }
        }
    }
}

/// Tests for Softmax function
#[cfg(test)]
mod softmax_tests {
    use super::*;

    #[test]
    fn test_softmax_forward() {
        let input_data = vec![1.0, 2.0, 3.0];
        let input = Tensor::new(input_data, vec![3]).unwrap();
        let general_input = GeneralTensor::Float(input);
        
        let mut softmax = Softmax::new();
        let output = softmax.run(&[&general_input], false).unwrap();
        
        // Softmax output should sum to 1
        let sum: f32 = output.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Softmax should sum to 1, got {}", sum);
        
        // All values should be positive
        for &val in output.data() {
            assert!(val > 0.0, "Softmax values should be positive, got {}", val);
        }
    }

    #[test]
    fn test_softmax_gradient() {
        let input_data = vec![0.5, -0.3, 1.2];
        let input = Tensor::new(input_data, vec![3]).unwrap();
        let general_input = GeneralTensor::Float(input);
        
        let mut softmax = Softmax::new();
        
        function_test_utils::numerical_gradient_check(
            &mut softmax,
            &[&general_input],
            1e-5,
            1e-3,
        ).unwrap();
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Test with large values that could cause overflow
        let input_data = vec![100.0, 101.0, 102.0];
        let input = Tensor::new(input_data, vec![3]).unwrap();
        let general_input = GeneralTensor::Float(input);
        
        let mut softmax = Softmax::new();
        let output = softmax.run(&[&general_input], false).unwrap();
        
        // Should still sum to 1 and be finite
        let sum: f32 = output.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        for &val in output.data() {
            assert!(val.is_finite());
        }
    }
}

/// Tests for Matrix Multiplication
#[cfg(test)]
mod matmul_tests {
    use super::*;

    #[test]
    fn test_matmul_forward() {
        // Test 2x3 * 3x2 = 2x2
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        
        let a = Tensor::new(a_data, vec![2, 3]).unwrap();
        let b = Tensor::new(b_data, vec![3, 2]).unwrap();
        let general_a = GeneralTensor::Float(a);
        let general_b = GeneralTensor::Float(b);
        
        let mut matmul = MatMul::new();
        let output = matmul.run(&[&general_a, &general_b], false).unwrap();
        
        assert_tensor_shape!(output, vec![2, 2]);
        
        // Expected result: [[4, 5], [10, 11]]
        let expected = vec![4.0, 5.0, 10.0, 11.0];
        assert_eq!(output.data(), &expected);
    }

    #[test]
    fn test_matmul_gradient() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![0.5, -0.5, 1.0, 0.0];
        
        let a = Tensor::new(a_data, vec![2, 2]).unwrap();
        let b = Tensor::new(b_data, vec![2, 2]).unwrap();
        let general_a = GeneralTensor::Float(a);
        let general_b = GeneralTensor::Float(b);
        
        let mut matmul = MatMul::new();
        
        function_test_utils::numerical_gradient_check(
            &mut matmul,
            &[&general_a, &general_b],
            1e-5,
            1e-3,
        ).unwrap();
    }

    #[test]
    fn test_matmul_incompatible_shapes() {
        let a_data = vec![1.0, 2.0];
        let b_data = vec![1.0, 2.0, 3.0];
        
        let a = Tensor::new(a_data, vec![2]).unwrap();
        let b = Tensor::new(b_data, vec![3]).unwrap();
        let general_a = GeneralTensor::Float(a);
        let general_b = GeneralTensor::Float(b);
        
        let mut matmul = MatMul::new();
        
        // Should fail due to incompatible shapes
        assert!(matmul.run(&[&general_a, &general_b], false).is_err());
    }

    proptest! {
        #[test]
        fn test_matmul_properties(
            m in 1..10usize,
            n in 1..10usize,
            k in 1..10usize,
        ) {
            let a_data = create_random_tensor(&[m, k], &mut thread_rng());
            let b_data = create_random_tensor(&[k, n], &mut thread_rng());
            
            let a = Tensor::new(a_data, vec![m, k]).unwrap();
            let b = Tensor::new(b_data, vec![k, n]).unwrap();
            let general_a = GeneralTensor::Float(a);
            let general_b = GeneralTensor::Float(b);
            
            let mut matmul = MatMul::new();
            let output = matmul.run(&[&general_a, &general_b], false).unwrap();
            
            // Output shape should be [m, n]
            assert_tensor_shape!(output, vec![m, n]);
            
            // All values should be finite
            for &val in output.data() {
                assert!(val.is_finite());
            }
        }
    }
}

/// Tests for Add function
#[cfg(test)]
mod add_tests {
    use super::*;

    #[test]
    fn test_add_forward() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![0.5, 1.5, 2.5, 3.5];
        
        let a = Tensor::new(a_data, vec![2, 2]).unwrap();
        let b = Tensor::new(b_data, vec![2, 2]).unwrap();
        let general_a = GeneralTensor::Float(a);
        let general_b = GeneralTensor::Float(b);
        
        let mut add = Add::new();
        let output = add.run(&[&general_a, &general_b], false).unwrap();
        
        let expected = vec![1.5, 3.5, 5.5, 7.5];
        assert_eq!(output.data(), &expected);
    }

    #[test]
    fn test_add_gradient() {
        let a_data = vec![1.0, 2.0];
        let b_data = vec![3.0, 4.0];
        
        let a = Tensor::new(a_data, vec![2]).unwrap();
        let b = Tensor::new(b_data, vec![2]).unwrap();
        let general_a = GeneralTensor::Float(a);
        let general_b = GeneralTensor::Float(b);
        
        let mut add = Add::new();
        let output = add.run(&[&general_a, &general_b], true).unwrap();
        let out_grad = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        
        let grads = add.grad(&[&general_a, &general_b], &out_grad).unwrap();
        
        // Gradient of addition is just the output gradient passed through
        assert_eq!(grads[0].data(), &[1.0, 2.0]);
        assert_eq!(grads[1].data(), &[1.0, 2.0]);
    }
}

/// Tests for Embedding function
#[cfg(test)]
mod embedding_tests {
    use super::*;

    #[test]
    fn test_embedding_forward() {
        let vocab_size = 5;
        let embedding_dim = 3;
        
        // Create embedding weights
        let weights_data = (0..vocab_size * embedding_dim).map(|i| i as f32).collect();
        let weights = Tensor::new(weights_data, vec![vocab_size, embedding_dim]).unwrap();
        let general_weights = GeneralTensor::Float(weights);
        
        // Create indices
        let indices_data = vec![0, 2, 1];
        let indices = Tensor::new(indices_data, vec![3]).unwrap();
        let general_indices = GeneralTensor::Usize(indices);
        
        let mut embedding = Embedding::new();
        let output = embedding.run(&[&general_weights, &general_indices], false).unwrap();
        
        assert_tensor_shape!(output, vec![3, embedding_dim]);
        
        // Check that we get the correct embeddings
        let expected = vec![
            0.0, 1.0, 2.0,  // Row 0
            6.0, 7.0, 8.0,  // Row 2
            3.0, 4.0, 5.0,  // Row 1
        ];
        assert_eq!(output.data(), &expected);
    }

    #[test]
    fn test_embedding_gradient() {
        let vocab_size = 3;
        let embedding_dim = 2;
        
        let weights_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights = Tensor::new(weights_data, vec![vocab_size, embedding_dim]).unwrap();
        let general_weights = GeneralTensor::Float(weights);
        
        let indices_data = vec![0, 2];
        let indices = Tensor::new(indices_data, vec![2]).unwrap();
        let general_indices = GeneralTensor::Usize(indices);
        
        let mut embedding = Embedding::new();
        let output = embedding.run(&[&general_weights, &general_indices], true).unwrap();
        let out_grad = Tensor::ones(output.shape());
        
        let grads = embedding.grad(&[&general_weights, &general_indices], &out_grad).unwrap();
        
        // Gradient should accumulate at the selected indices
        assert_tensor_shape!(grads[0], vec![vocab_size, embedding_dim]);
    }
}

/// Tests for Layer Normalization
#[cfg(test)]
mod layer_norm_tests {
    use super::*;

    #[test]
    fn test_layer_norm_forward() {
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::new(input_data, vec![2, 3]).unwrap();
        let general_input = GeneralTensor::Float(input);
        
        let mut layer_norm = LayerNorm::new();
        let output = layer_norm.run(&[&general_input], false).unwrap();
        
        assert_tensor_shape!(output, vec![2, 3]);
        
        // Each row should have approximately zero mean and unit variance
        for row in 0..2 {
            let row_data: Vec<f32> = (0..3)
                .map(|col| output.get(&[row, col]).unwrap().clone())
                .collect();
            
            let mean: f32 = row_data.iter().sum::<f32>() / row_data.len() as f32;
            let variance: f32 = row_data.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / row_data.len() as f32;
            
            assert!(mean.abs() < 1e-6, "LayerNorm mean should be ~0, got {}", mean);
            assert!((variance - 1.0).abs() < 1e-6, "LayerNorm variance should be ~1, got {}", variance);
        }
    }

    #[test]
    fn test_layer_norm_gradient() {
        let input_data = vec![0.5, -0.3, 1.2, -0.8];
        let input = Tensor::new(input_data, vec![1, 4]).unwrap();
        let general_input = GeneralTensor::Float(input);
        
        let mut layer_norm = LayerNorm::new();
        
        function_test_utils::numerical_gradient_check(
            &mut layer_norm,
            &[&general_input],
            1e-5,
            1e-3,
        ).unwrap();
    }
}

/// Tests for Cross-Entropy Loss
#[cfg(test)]
mod crossentropy_tests {
    use super::*;

    #[test]
    fn test_crossentropy_forward() {
        // Logits for 2 samples, 3 classes
        let logits_data = vec![1.0, 2.0, 0.5, 0.8, 1.5, 0.2];
        let logits = Tensor::new(logits_data, vec![2, 3]).unwrap();
        let general_logits = GeneralTensor::Float(logits);
        
        // Target classes
        let targets_data = vec![1, 2]; // Class indices
        let targets = Tensor::new(targets_data, vec![2]).unwrap();
        let general_targets = GeneralTensor::Usize(targets);
        
        let mut crossentropy = CrossEntropy::new();
        let output = crossentropy.run(&[&general_logits, &general_targets], false).unwrap();
        
        // Output should be a scalar (average loss)
        assert_tensor_shape!(output, vec![1]);
        
        // Loss should be positive
        assert!(output.data()[0] > 0.0);
    }

    #[test]
    fn test_crossentropy_gradient() {
        let logits_data = vec![0.5, -0.3, 1.2];
        let logits = Tensor::new(logits_data, vec![1, 3]).unwrap();
        let general_logits = GeneralTensor::Float(logits);
        
        let targets_data = vec![1];
        let targets = Tensor::new(targets_data, vec![1]).unwrap();
        let general_targets = GeneralTensor::Usize(targets);
        
        let mut crossentropy = CrossEntropy::new();
        
        // Note: We only check gradient w.r.t. logits, not targets
        function_test_utils::numerical_gradient_check(
            &mut crossentropy,
            &[&general_logits, &general_targets],
            1e-5,
            1e-3,
        ).unwrap();
    }
}

/// Tests for Dropout
#[cfg(test)]
mod dropout_tests {
    use super::*;

    #[test]
    fn test_dropout_training_mode() {
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input = Tensor::new(input_data.clone(), vec![5]).unwrap();
        let general_input = GeneralTensor::Float(input);
        
        let dropout_rate = 0.5;
        let mut dropout = Dropout::new(dropout_rate);
        
        // In training mode, some values should be zeroed out
        let output = dropout.run(&[&general_input], true).unwrap();
        
        assert_tensor_shape!(output, vec![5]);
        
        // Count how many values are non-zero
        let non_zero_count = output.data().iter().filter(|&&x| x != 0.0).count();
        
        // With dropout rate 0.5, we expect roughly half the values to be non-zero
        // (though this is probabilistic, so we use a loose bound)
        assert!(non_zero_count >= 1 && non_zero_count <= 4);
    }

    #[test]
    fn test_dropout_inference_mode() {
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input = Tensor::new(input_data.clone(), vec![5]).unwrap();
        let general_input = GeneralTensor::Float(input);
        
        let dropout_rate = 0.5;
        let mut dropout = Dropout::new(dropout_rate);
        
        // In inference mode, all values should be preserved (scaled)
        let output = dropout.run(&[&general_input], false).unwrap();
        
        assert_tensor_shape!(output, vec![5]);
        
        // All values should be scaled by (1 - dropout_rate)
        let expected_scale = 1.0 - dropout_rate;
        for (i, (&input_val, &output_val)) in input_data.iter().zip(output.data().iter()).enumerate() {
            let expected = input_val * expected_scale;
            assert!((output_val - expected).abs() < 1e-6, 
                "Dropout inference scaling incorrect at index {}: expected {}, got {}", 
                i, expected, output_val);
        }
    }
}

/// Integration tests for function combinations
#[cfg(test)]
mod function_integration_tests {
    use super::*;

    #[test]
    fn test_mlp_layer_simulation() {
        // Simulate a simple MLP layer: input -> linear -> relu -> output
        let input_data = vec![-1.0, 0.5, 2.0, -0.3];
        let input = Tensor::new(input_data, vec![2, 2]).unwrap();
        let general_input = GeneralTensor::Float(input);
        
        // Weight matrix for linear transformation
        let weight_data = vec![0.5, -0.3, 1.2, 0.8];
        let weight = Tensor::new(weight_data, vec![2, 2]).unwrap();
        let general_weight = GeneralTensor::Float(weight);
        
        // Linear transformation (matrix multiplication)
        let mut matmul = MatMul::new();
        let linear_output = matmul.run(&[&general_input, &general_weight], true).unwrap();
        let general_linear = GeneralTensor::Float(linear_output);
        
        // Apply ReLU activation
        let mut relu = ReLU::new();
        let final_output = relu.run(&[&general_linear], true).unwrap();
        
        // Verify shapes are preserved
        assert_tensor_shape!(final_output, vec![2, 2]);
        
        // Verify ReLU property (no negative values)
        for &val in final_output.data() {
            assert!(val >= 0.0, "ReLU output should be non-negative, got {}", val);
        }
    }

    #[test]
    fn test_attention_mechanism_components() {
        // Test components that would be used in attention mechanism
        let seq_len = 3;
        let d_model = 4;
        
        // Input sequence
        let input_data = (0..seq_len * d_model).map(|i| i as f32 * 0.1).collect();
        let input = Tensor::new(input_data, vec![seq_len, d_model]).unwrap();
        let general_input = GeneralTensor::Float(input);
        
        // Query, Key, Value weight matrices
        let qkv_data = create_random_tensor(&[d_model, d_model], &mut StdRng::seed_from_u64(42));
        let qkv_weights = Tensor::new(qkv_data, vec![d_model, d_model]).unwrap();
        let general_qkv = GeneralTensor::Float(qkv_weights);
        
        // Compute Q, K, V
        let mut matmul = MatMul::new();
        let q = matmul.run(&[&general_input, &general_qkv], true).unwrap();
        let k = matmul.run(&[&general_input, &general_qkv], true).unwrap();
        let v = matmul.run(&[&general_input, &general_qkv], true).unwrap();
        
        // Verify shapes
        assert_tensor_shape!(q, vec![seq_len, d_model]);
        assert_tensor_shape!(k, vec![seq_len, d_model]);
        assert_tensor_shape!(v, vec![seq_len, d_model]);
        
        // Apply softmax to attention scores (simplified)
        let general_q = GeneralTensor::Float(q);
        let mut softmax = Softmax::new();
        let attention_weights = softmax.run(&[&general_q], true).unwrap();
        
        // Each row should sum to approximately 1 after softmax
        for row in 0..seq_len {
            let row_sum: f32 = (0..d_model)
                .map(|col| *attention_weights.get(&[row, col]).unwrap())
                .sum();
            assert!((row_sum - 1.0).abs() < 1e-5, "Softmax row should sum to 1, got {}", row_sum);
        }
    }
}

/// Performance tests for functions
#[cfg(test)]
mod function_performance_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_large_matmul_performance() {
        let size = 100;
        let a_data = create_ones_tensor(&[size, size]);
        let b_data = create_ones_tensor(&[size, size]);
        
        let a = Tensor::new(a_data, vec![size, size]).unwrap();
        let b = Tensor::new(b_data, vec![size, size]).unwrap();
        let general_a = GeneralTensor::Float(a);
        let general_b = GeneralTensor::Float(b);
        
        let mut matmul = MatMul::new();
        
        let _result = assert_performance!(
            matmul.run(&[&general_a, &general_b], false).unwrap(),
            Duration::from_millis(200),
            "Large matrix multiplication"
        );
    }

    #[test]
    fn test_softmax_performance() {
        let size = 10000;
        let input_data = create_random_tensor(&[size], &mut thread_rng());
        let input = Tensor::new(input_data, vec![size]).unwrap();
        let general_input = GeneralTensor::Float(input);
        
        let mut softmax = Softmax::new();
        
        let _result = assert_performance!(
            softmax.run(&[&general_input], false).unwrap(),
            Duration::from_millis(50),
            "Large softmax computation"
        );
    }
}