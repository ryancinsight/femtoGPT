use femto_gpt::tensor::{GeneralTensor, Tensor, TensorElement, TensorError};
use proptest::prelude::*;
use rand::prelude::*;
use std::collections::HashMap;

// Import test utilities
#[path = "../common/mod.rs"]
mod common;
use common::*;

/// Unit tests for tensor element trait implementations
#[cfg(test)]
mod tensor_element_tests {
    use super::*;

    #[test]
    fn test_f32_element_zero_one() {
        assert_eq!(f32::zero(), 0.0);
        assert_eq!(f32::one(), 1.0);
    }

    #[test]
    fn test_usize_element_zero_one() {
        assert_eq!(usize::zero(), 0);
        assert_eq!(usize::one(), 1);
    }
}

/// Unit tests for tensor creation and basic operations
#[cfg(test)]
mod tensor_creation_tests {
    use super::*;

    #[test]
    fn test_tensor_new_with_shape() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = Tensor::new(data.clone(), shape.clone()).unwrap();
        
        assert_eq!(tensor.data(), &data);
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.size(), 4);
    }

    #[test]
    fn test_tensor_new_with_mismatched_shape() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = vec![2, 2]; // Size 4, but data has 3 elements
        
        assert!(Tensor::new(data, shape).is_err());
    }

    #[test]
    fn test_tensor_zeros() {
        let shape = vec![3, 2];
        let tensor = Tensor::<f32>::zeros(&shape);
        
        assert_tensor_shape!(tensor, shape);
        assert_eq!(tensor.size(), 6);
        for &val in tensor.data() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_tensor_ones() {
        let shape = vec![2, 3];
        let tensor = Tensor::<f32>::ones(&shape);
        
        assert_tensor_shape!(tensor, shape);
        assert_eq!(tensor.size(), 6);
        for &val in tensor.data() {
            assert_eq!(val, 1.0);
        }
    }

    #[test]
    fn test_tensor_randn() {
        let mut rng = StdRng::seed_from_u64(42);
        let shape = vec![10, 10];
        let tensor = Tensor::<f32>::randn(&mut rng, &shape);
        
        assert_tensor_shape!(tensor, shape);
        assert_eq!(tensor.size(), 100);
        
        // Check that values are roughly normally distributed
        let data = tensor.data();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let variance: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        
        // With enough samples, mean should be close to 0 and variance close to 1
        assert!((mean.abs()) < 0.3, "Mean too far from 0: {}", mean);
        assert!((variance - 1.0).abs() < 0.3, "Variance too far from 1: {}", variance);
    }

    #[test]
    fn test_tensor_uniform() {
        let mut rng = StdRng::seed_from_u64(42);
        let shape = vec![5, 5];
        let low = -2.0;
        let high = 3.0;
        let tensor = Tensor::<f32>::uniform(&mut rng, &shape, low, high);
        
        assert_tensor_shape!(tensor, shape);
        
        // Check that all values are within bounds
        for &val in tensor.data() {
            assert!(val >= low && val < high, "Value {} not in range [{}, {})", val, low, high);
        }
    }

    #[test]
    fn test_empty_tensor() {
        let data: Vec<f32> = vec![];
        let shape = vec![0];
        let tensor = Tensor::new(data, shape.clone()).unwrap();
        
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.size(), 0);
        assert!(tensor.data().is_empty());
    }

    proptest! {
        #[test]
        fn test_tensor_creation_property(
            shape in small_tensor_shape_strategy(),
        ) {
            let size = shape.iter().product::<usize>();
            let data = create_random_tensor(&shape, &mut thread_rng());
            
            let tensor = Tensor::new(data.clone(), shape.clone()).unwrap();
            
            assert_eq!(tensor.shape(), &shape);
            assert_eq!(tensor.size(), size);
            assert_eq!(tensor.data(), &data);
        }
    }
}

/// Unit tests for tensor indexing and access
#[cfg(test)]
mod tensor_access_tests {
    use super::*;

    #[test]
    fn test_tensor_get_valid_index() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let tensor = Tensor::new(data, shape).unwrap();
        
        assert_eq!(tensor.get(&[0, 0]).unwrap(), &1.0);
        assert_eq!(tensor.get(&[0, 1]).unwrap(), &2.0);
        assert_eq!(tensor.get(&[0, 2]).unwrap(), &3.0);
        assert_eq!(tensor.get(&[1, 0]).unwrap(), &4.0);
        assert_eq!(tensor.get(&[1, 1]).unwrap(), &5.0);
        assert_eq!(tensor.get(&[1, 2]).unwrap(), &6.0);
    }

    #[test]
    fn test_tensor_get_invalid_index() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = Tensor::new(data, shape).unwrap();
        
        // Out of bounds indices
        assert!(tensor.get(&[2, 0]).is_err());
        assert!(tensor.get(&[0, 2]).is_err());
        assert!(tensor.get(&[2, 2]).is_err());
        
        // Wrong number of dimensions
        assert!(tensor.get(&[0]).is_err());
        assert!(tensor.get(&[0, 0, 0]).is_err());
    }

    #[test]
    fn test_tensor_get_mut_valid_index() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let mut tensor = Tensor::new(data, shape).unwrap();
        
        *tensor.get_mut(&[0, 0]).unwrap() = 10.0;
        *tensor.get_mut(&[1, 1]).unwrap() = 20.0;
        
        assert_eq!(tensor.get(&[0, 0]).unwrap(), &10.0);
        assert_eq!(tensor.get(&[1, 1]).unwrap(), &20.0);
    }

    proptest! {
        #[test]
        fn test_tensor_access_property(
            shape in small_tensor_shape_strategy(),
        ) {
            let size = shape.iter().product::<usize>();
            if size == 0 { return Ok(()); }
            
            let data = create_sequential_tensor(&shape);
            let tensor = Tensor::new(data, shape.clone()).unwrap();
            
            // Test that we can access all valid indices
            let mut indices = vec![0; shape.len()];
            for i in 0..size {
                // Convert linear index to multi-dimensional index
                let mut temp = i;
                for (dim_idx, &dim_size) in shape.iter().enumerate().rev() {
                    indices[dim_idx] = temp % dim_size;
                    temp /= dim_size;
                }
                
                // Should be able to access this index
                assert!(tensor.get(&indices).is_ok());
            }
        }
    }
}

/// Unit tests for tensor operations
#[cfg(test)]
mod tensor_operations_tests {
    use super::*;

    #[test]
    fn test_tensor_add() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![5.0, 6.0, 7.0, 8.0];
        let shape = vec![2, 2];
        
        let tensor1 = Tensor::new(data1, shape.clone()).unwrap();
        let tensor2 = Tensor::new(data2, shape.clone()).unwrap();
        
        let result = (&tensor1 + &tensor2).unwrap();
        let expected = vec![6.0, 8.0, 10.0, 12.0];
        
        assert_tensor_shape!(result, shape);
        assert_eq!(result.data(), &expected);
    }

    #[test]
    fn test_tensor_add_mismatched_shapes() {
        let tensor1 = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let tensor2 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        
        assert!((&tensor1 + &tensor2).is_err());
    }

    #[test]
    fn test_tensor_sub() {
        let data1 = vec![5.0, 6.0, 7.0, 8.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        
        let tensor1 = Tensor::new(data1, shape.clone()).unwrap();
        let tensor2 = Tensor::new(data2, shape.clone()).unwrap();
        
        let result = (&tensor1 - &tensor2).unwrap();
        let expected = vec![4.0, 4.0, 4.0, 4.0];
        
        assert_tensor_shape!(result, shape);
        assert_eq!(result.data(), &expected);
    }

    #[test]
    fn test_tensor_mul() {
        let data1 = vec![2.0, 3.0, 4.0, 5.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        
        let tensor1 = Tensor::new(data1, shape.clone()).unwrap();
        let tensor2 = Tensor::new(data2, shape.clone()).unwrap();
        
        let result = (&tensor1 * &tensor2).unwrap();
        let expected = vec![2.0, 6.0, 12.0, 20.0];
        
        assert_tensor_shape!(result, shape);
        assert_eq!(result.data(), &expected);
    }

    #[test]
    fn test_tensor_scalar_mul() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = Tensor::new(data, shape.clone()).unwrap();
        
        let result = &tensor * 2.0;
        let expected = vec![2.0, 4.0, 6.0, 8.0];
        
        assert_tensor_shape!(result, shape);
        assert_eq!(result.data(), &expected);
    }

    proptest! {
        #[test]
        fn test_tensor_operations_property(
            shape in small_tensor_shape_strategy(),
            scalar in -10.0..10.0f32,
        ) {
            let size = shape.iter().product::<usize>();
            if size == 0 { return Ok(()); }
            
            let data1 = create_random_tensor(&shape, &mut thread_rng());
            let data2 = create_random_tensor(&shape, &mut thread_rng());
            
            let tensor1 = Tensor::new(data1, shape.clone()).unwrap();
            let tensor2 = Tensor::new(data2, shape.clone()).unwrap();
            
            // Test commutativity of addition
            let add1 = (&tensor1 + &tensor2).unwrap();
            let add2 = (&tensor2 + &tensor1).unwrap();
            assert_tensor_eq!(add1, add2, 1e-6);
            
            // Test associativity with scalar multiplication
            let mul1 = (&tensor1 * scalar) * 2.0;
            let mul2 = &tensor1 * (scalar * 2.0);
            assert_tensor_eq!(mul1, mul2, 1e-6);
            
            // Test distributivity
            let dist1 = (&tensor1 + &tensor2) * scalar;
            let dist2 = (&tensor1 * scalar) + (&tensor2 * scalar);
            assert_tensor_eq!(dist1, dist2, 1e-5);
        }
    }
}

/// Unit tests for tensor reshaping and views
#[cfg(test)]
mod tensor_reshape_tests {
    use super::*;

    #[test]
    fn test_tensor_reshape_valid() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(data.clone(), vec![2, 3]).unwrap();
        
        let reshaped = tensor.reshape(&[3, 2]).unwrap();
        
        assert_tensor_shape!(reshaped, vec![3, 2]);
        assert_eq!(reshaped.data(), &data); // Data should remain the same
    }

    #[test]
    fn test_tensor_reshape_invalid_size() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, vec![2, 2]).unwrap();
        
        // Try to reshape to incompatible size
        assert!(tensor.reshape(&[2, 3]).is_err()); // 4 elements can't fit in 2x3
    }

    #[test]
    fn test_tensor_transpose_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(data, vec![2, 3]).unwrap();
        
        let transposed = tensor.transpose().unwrap();
        let expected_data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        
        assert_tensor_shape!(transposed, vec![3, 2]);
        assert_eq!(transposed.data(), &expected_data);
    }

    #[test]
    fn test_tensor_transpose_1d_error() {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::new(data, vec![3]).unwrap();
        
        // 1D tensors can't be transposed
        assert!(tensor.transpose().is_err());
    }

    proptest! {
        #[test]
        fn test_tensor_reshape_property(
            original_shape in small_tensor_shape_strategy(),
        ) {
            let size = original_shape.iter().product::<usize>();
            if size == 0 { return Ok(()); }
            
            let data = create_sequential_tensor(&original_shape);
            let tensor = Tensor::new(data.clone(), original_shape).unwrap();
            
            // Generate a compatible reshape
            let factors = get_factors(size);
            if factors.len() >= 2 {
                let new_shape = vec![factors[0], size / factors[0]];
                let reshaped = tensor.reshape(&new_shape).unwrap();
                
                assert_eq!(reshaped.size(), size);
                assert_eq!(reshaped.data(), &data);
                assert_tensor_shape!(reshaped, new_shape);
            }
        }
    }

    // Helper function to get factors of a number
    fn get_factors(n: usize) -> Vec<usize> {
        let mut factors = Vec::new();
        for i in 1..=n {
            if n % i == 0 {
                factors.push(i);
            }
        }
        factors
    }
}

/// Unit tests for GeneralTensor enum
#[cfg(test)]
mod general_tensor_tests {
    use super::*;

    #[test]
    fn test_general_tensor_float() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let general = GeneralTensor::Float(tensor.clone());
        
        assert_eq!(general.size(), 3);
        assert_eq!(general.shape(), &[3]);
        assert!(general.as_float().is_ok());
        assert!(general.as_usize().is_err());
        
        let float_tensor = general.as_float().unwrap();
        assert_eq!(float_tensor.data(), tensor.data());
    }

    #[test]
    fn test_general_tensor_usize() {
        let tensor = Tensor::new(vec![1, 2, 3], vec![3]).unwrap();
        let general = GeneralTensor::Usize(tensor.clone());
        
        assert_eq!(general.size(), 3);
        assert_eq!(general.shape(), &[3]);
        assert!(general.as_usize().is_ok());
        assert!(general.as_float().is_err());
        
        let usize_tensor = general.as_usize().unwrap();
        assert_eq!(usize_tensor.data(), tensor.data());
    }
}

/// Performance tests for tensor operations
#[cfg(test)]
mod tensor_performance_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_large_tensor_creation_performance() {
        let shape = vec![1000, 1000];
        
        let result = assert_performance!(
            Tensor::<f32>::zeros(&shape),
            Duration::from_millis(100),
            "Large tensor creation"
        );
        
        assert_eq!(result.size(), 1_000_000);
    }

    #[test]
    fn test_tensor_addition_performance() {
        let shape = vec![500, 500];
        let tensor1 = Tensor::<f32>::ones(&shape);
        let tensor2 = Tensor::<f32>::ones(&shape);
        
        let _result = assert_performance!(
            (&tensor1 + &tensor2).unwrap(),
            Duration::from_millis(50),
            "Large tensor addition"
        );
    }
}

/// Error handling tests
#[cfg(test)]
mod tensor_error_tests {
    use super::*;

    #[test]
    fn test_tensor_error_types() {
        // Test UnexpectedType error
        let general = GeneralTensor::Float(Tensor::ones(&[2, 2]));
        assert_error!(general.as_usize(), TensorError::UnexpectedType);
        
        // Test UnexpectedShape error
        let data = vec![1.0, 2.0, 3.0];
        let shape = vec![2, 2]; // Wrong size
        assert_error!(Tensor::new(data, shape), TensorError::UnexpectedShape);
        
        // Test InvalidIndex error
        let tensor = Tensor::ones(&[2, 2]);
        assert_error!(tensor.get(&[2, 0]), TensorError::InvalidIndex);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_tensor_workflow_integration() {
        let mut rng = StdRng::seed_from_u64(42);
        
        // Create initial tensors
        let tensor1 = Tensor::<f32>::randn(&mut rng, &[4, 4]);
        let tensor2 = Tensor::<f32>::uniform(&mut rng, &[4, 4], -1.0, 1.0);
        
        // Perform operations
        let sum = (&tensor1 + &tensor2).unwrap();
        let product = (&sum * 2.0);
        let reshaped = product.reshape(&[8, 2]).unwrap();
        let transposed = reshaped.transpose().unwrap();
        
        // Verify final result
        assert_tensor_shape!(transposed, vec![2, 8]);
        assert_eq!(transposed.size(), 16);
        
        // Verify data integrity through operations
        assert!(transposed.data().iter().all(|&x| x.is_finite()));
    }
}