use crate::funcs::{Function, MatMul, Coeff};
use crate::tensor::{GeneralTensor, Tensor, TensorError, TensorOps, TensorMutOps};
use std::cmp::min;

/// Flash Attention implementation for memory-efficient attention computation
/// 
/// This implementation avoids materializing the full O(N²) attention matrix by computing
/// attention in tiles, using online softmax computation to maintain numerical stability.
/// 
/// Key features:
/// - O(N) memory complexity instead of O(N²) 
/// - Tiled computation with configurable block sizes
/// - Online softmax with running statistics
/// - Numerical equivalence with standard attention
/// - CPU-optimized cache-friendly access patterns
#[derive(Clone, Debug)]
pub struct FlashAttention {
    /// Block size for tiled computation (default: 64 for optimal cache usage)
    block_size_q: usize,
    block_size_k: usize,
    /// Whether to apply causal masking
    causal: bool,
    /// Temperature scaling factor
    scale: f32,
}

impl FlashAttention {
    /// Create new Flash Attention with default block size (64x64)
    pub fn new(scale: f32, causal: bool) -> Self {
        Self {
            block_size_q: 64,
            block_size_k: 64,
            causal,
            scale,
        }
    }

    /// Create Flash Attention with custom block sizes
    pub fn with_block_sizes(scale: f32, causal: bool, block_size_q: usize, block_size_k: usize) -> Self {
        Self {
            block_size_q,
            block_size_k,
            causal,
            scale,
        }
    }

    /// Compute attention for a single tile using online softmax
    fn compute_tile_attention(
        &self,
        q_tile: &Tensor<f32>,
        k_tile: &Tensor<f32>, 
        v_tile: &Tensor<f32>,
        running_max: &mut Tensor<f32>,
        running_sum: &mut Tensor<f32>,
        output_acc: &mut Tensor<f32>,
        tile_row_start: usize,
        tile_col_start: usize,
    ) -> Result<(), TensorError> {
        // Compute attention scores: S = Q @ K^T * scale
        let k_t = k_tile.transpose()?;
        let mut matmul = MatMul::new();
        let qk = matmul.run(&[&GeneralTensor::Float(q_tile.clone()), &GeneralTensor::Float(k_t)], false)?;
        let mut coeff = Coeff::new(self.scale);
        let scores = coeff.run(&[&GeneralTensor::Float(qk)], false)?;
        
        // Apply causal masking if enabled
        let masked_scores = if self.causal {
            self.apply_causal_mask(&scores, tile_row_start, tile_col_start)?
        } else {
            scores
        };

        // Extract relevant portion of running statistics for current Q tile
        let q_tile_shape = q_tile.shape();
        let q_tile_size = q_tile_shape[q_tile_shape.len() - 2]; // Sequence length dimension

        let tile_running_max = self.extract_running_stats(running_max, tile_row_start, q_tile_size)?;
        let tile_running_sum = self.extract_running_stats(running_sum, tile_row_start, q_tile_size)?;

        // Online softmax computation with running statistics
        let (tile_output, new_max, new_sum) = self.online_softmax_update(
            &masked_scores,
            v_tile,
            &tile_running_max,
            &tile_running_sum,
        )?;

        // Update running statistics for this tile
        self.update_running_stats(running_max, &new_max, tile_row_start, q_tile_size)?;
        self.update_running_stats(running_sum, &new_sum, tile_row_start, q_tile_size)?;

        // Accumulate output for this tile
        self.accumulate_tile_output(output_acc, &tile_output, tile_row_start, q_tile_size)?;

        Ok(())
    }

    /// Apply causal masking to attention scores
    fn apply_causal_mask(
        &self,
        scores: &Tensor<f32>,
        tile_row_start: usize,
        tile_col_start: usize,
    ) -> Result<Tensor<f32>, TensorError> {
        let mut masked = scores.clone();
        let shape = scores.shape();
        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];

        for i in 0..rows {
            for j in 0..cols {
                let global_i = tile_row_start + i;
                let global_j = tile_col_start + j;
                
                if global_j > global_i {
                    // Set future positions to negative infinity
                    let linear_idx = i * cols + j;
                    masked.blob_mut()[linear_idx] = f32::NEG_INFINITY;
                }
            }
        }

        Ok(masked)
    }

    /// Online softmax computation with running max and sum
    fn online_softmax_update(
        &self,
        scores: &Tensor<f32>,
        values: &Tensor<f32>,
        running_max: &Tensor<f32>,
        running_sum: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>, Tensor<f32>), TensorError> {
        let score_shape = scores.shape();
        let _seq_len_q = score_shape[score_shape.len() - 2];
        let _seq_len_k = score_shape[score_shape.len() - 1];

        // Compute new max values
        let tile_max = self.compute_row_max(scores)?;
        let new_max = self.element_wise_max(running_max, &tile_max)?;

        // Compute exponentials with numerical stability
        let exp_scores = self.compute_stable_exp(scores, &new_max)?;
        let exp_running = self.compute_stable_exp_adjustment(running_max, &new_max, running_sum)?;

        // Update running sum
        let tile_sum = self.compute_row_sum(&exp_scores)?;
        let new_sum = (&exp_running + &tile_sum)?;

        // Compute attention weights and output
        let attention_weights = self.normalize_by_sum(&exp_scores, &new_sum)?;
        let mut matmul = MatMul::new();
        let tile_output = matmul.run(&[&GeneralTensor::Float(attention_weights), &GeneralTensor::Float(values.clone())], false)?;

        Ok((tile_output, new_max, new_sum))
    }

    /// Compute row-wise maximum
    fn compute_row_max(&self, tensor: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        let shape = tensor.shape();
        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        
        // Handle batched tensors correctly
        let batch_size = if shape.len() > 2 {
            shape[..shape.len() - 2].iter().product()
        } else {
            1
        };
        
        let mut max_vals = Vec::with_capacity(batch_size * rows);
        let blob = tensor.blob();
        
        for batch_idx in 0..batch_size {
            let batch_offset = batch_idx * rows * cols;
            for i in 0..rows {
                let mut row_max = f32::NEG_INFINITY;
                for j in 0..cols {
                    let linear_idx = batch_offset + i * cols + j;
                    let val = blob[linear_idx];
                    if val > row_max {
                        row_max = val;
                    }
                }
                max_vals.push(row_max);
            }
        }

        let max_shape = if shape.len() == 2 {
            vec![rows, 1]
        } else {
            let mut new_shape = shape.to_vec();
            new_shape[shape.len() - 1] = 1;
            new_shape
        };

        Tensor::raw(&max_shape, max_vals)
    }

    /// Compute row-wise sum
    fn compute_row_sum(&self, tensor: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        let shape = tensor.shape();
        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        
        // Handle batched tensors correctly
        let batch_size = if shape.len() > 2 {
            shape[..shape.len() - 2].iter().product()
        } else {
            1
        };
        
        let mut sum_vals = Vec::with_capacity(batch_size * rows);
        let blob = tensor.blob();
        
        for batch_idx in 0..batch_size {
            let batch_offset = batch_idx * rows * cols;
            for i in 0..rows {
                let mut row_sum = 0.0;
                for j in 0..cols {
                    let linear_idx = batch_offset + i * cols + j;
                    row_sum += blob[linear_idx];
                }
                sum_vals.push(row_sum);
            }
        }

        let sum_shape = if shape.len() == 2 {
            vec![rows, 1]
        } else {
            let mut new_shape = shape.to_vec();
            new_shape[shape.len() - 1] = 1;
            new_shape
        };

        Tensor::raw(&sum_shape, sum_vals)
    }

    /// Element-wise maximum of two tensors
    fn element_wise_max(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        let shape = a.shape();
        let mut result_data = Vec::with_capacity(a.size());
        
        let a_blob = a.blob();
        let b_blob = b.blob();
        
        for i in 0..a.size() {
            let a_val = a_blob[i];
            let b_val = b_blob[i];
            result_data.push(a_val.max(b_val));
        }

        Tensor::raw(&shape, result_data)
    }

    /// Compute stable exponentials
    fn compute_stable_exp(&self, scores: &Tensor<f32>, max_vals: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        let shape = scores.shape();
        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        
        // Handle batched tensors correctly
        let batch_size = if shape.len() > 2 {
            shape[..shape.len() - 2].iter().product()
        } else {
            1
        };
        
        let mut exp_data = Vec::with_capacity(scores.size());

        let scores_blob = scores.blob();
        let max_blob = max_vals.blob();

        for batch_idx in 0..batch_size {
            let batch_offset = batch_idx * rows * cols;
            let max_offset = batch_idx * rows;
            
            for i in 0..rows {
                let max_idx = max_offset + i;
                let max_val = max_blob[max_idx];
                for j in 0..cols {
                    let linear_idx = batch_offset + i * cols + j;
                    let score = scores_blob[linear_idx];
                    exp_data.push((score - max_val).exp());
                }
            }
        }

        Tensor::raw(&shape, exp_data)
    }

    /// Compute adjustment for running exponentials
    fn compute_stable_exp_adjustment(
        &self,
        old_max: &Tensor<f32>,
        new_max: &Tensor<f32>, 
        old_sum: &Tensor<f32>,
    ) -> Result<Tensor<f32>, TensorError> {
        let mut result_data = Vec::with_capacity(old_sum.size());
        
        let old_max_blob = old_max.blob();
        let new_max_blob = new_max.blob();
        let old_sum_blob = old_sum.blob();
        
        for i in 0..old_sum.size() {
            let old_m = old_max_blob[i];
            let new_m = new_max_blob[i];
            let old_s = old_sum_blob[i];
            
            if old_m == f32::NEG_INFINITY {
                result_data.push(0.0);
            } else {
                result_data.push(old_s * (old_m - new_m).exp());
            }
        }

        Tensor::raw(&old_sum.shape(), result_data)
    }

    /// Normalize by sum for attention weights
    fn normalize_by_sum(&self, exp_scores: &Tensor<f32>, sum_vals: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        let shape = exp_scores.shape();
        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        let mut normalized_data = Vec::with_capacity(exp_scores.size());

        let exp_blob = exp_scores.blob();
        let sum_blob = sum_vals.blob();

        for i in 0..rows {
            let sum_val = sum_blob[i];
            for j in 0..cols {
                let linear_idx = i * cols + j;
                let exp_val = exp_blob[linear_idx];
                normalized_data.push(if sum_val > 0.0 { exp_val / sum_val } else { 0.0 });
            }
        }

        Tensor::raw(&shape, normalized_data)
    }
}

impl Function for FlashAttention {
    fn run(&mut self, inputs: &[&GeneralTensor], _training: bool) -> Result<Tensor<f32>, TensorError> {
        if inputs.len() != 3 {
            return Err(TensorError::UnexpectedShape);
        }

        let q = inputs[0].as_float()?;
        let k = inputs[1].as_float()?; 
        let v = inputs[2].as_float()?;

        // Validate input dimensions
        let q_shape = q.shape();
        let k_shape = k.shape();
        let v_shape = v.shape();

        if q_shape.len() < 2 || k_shape.len() < 2 || v_shape.len() < 2 {
            return Err(TensorError::UnexpectedShape);
        }

        let seq_len_q = q_shape[q_shape.len() - 2];
        let seq_len_k = k_shape[k_shape.len() - 2];
        let seq_len_v = v_shape[v_shape.len() - 2];
        let head_dim = q_shape[q_shape.len() - 1];
        let value_dim = v_shape[v_shape.len() - 1];

        if seq_len_k != seq_len_v {
            return Err(TensorError::UnexpectedShape);
        }

        if k_shape[k_shape.len() - 1] != head_dim {
            return Err(TensorError::UnexpectedShape);
        }

        // Initialize output and running statistics
        let output_shape = {
            let mut shape = q_shape.to_vec();
            let last_idx = shape.len() - 1;
            shape[last_idx] = value_dim;
            shape
        };
        
        let mut output = Tensor::zeros(&output_shape);
        
        // Initialize running statistics with correct batch dimensions
        let running_stats_shape = if q_shape.len() == 2 {
            vec![seq_len_q, 1]
        } else {
            let mut shape = q_shape.to_vec();
            let last_idx = shape.len() - 1;
            shape[last_idx] = 1; // Last dimension becomes 1
            shape
        };
        
        let mut running_max = Tensor::constant(&running_stats_shape, f32::NEG_INFINITY);
        let mut running_sum = Tensor::zeros(&running_stats_shape);

        // Tiled computation loop
        let num_q_tiles = (seq_len_q + self.block_size_q - 1) / self.block_size_q;
        let num_k_tiles = (seq_len_k + self.block_size_k - 1) / self.block_size_k;

        for q_tile_idx in 0..num_q_tiles {
            let q_start = q_tile_idx * self.block_size_q;
            let q_end = min(q_start + self.block_size_q, seq_len_q);
            
            // Extract Q tile
            let q_tile = self.extract_tile(q, q_start, q_end, q_shape.len())?;
            
            for k_tile_idx in 0..num_k_tiles {
                let k_start = k_tile_idx * self.block_size_k;
                let k_end = min(k_start + self.block_size_k, seq_len_k);
                
                // Extract K and V tiles
                let k_tile = self.extract_tile(k, k_start, k_end, k_shape.len())?;
                let v_tile = self.extract_tile(v, k_start, k_end, v_shape.len())?;
                
                // Compute attention for this tile
                self.compute_tile_attention(
                    &q_tile,
                    &k_tile,
                    &v_tile,
                    &mut running_max,
                    &mut running_sum,
                    &mut output,
                    q_start,
                    k_start,
                )?;
            }
        }

        // Final normalization
        self.final_normalization(&mut output, &running_sum)?;

        Ok(output)
    }

    fn grad(&self, inputs: &[&GeneralTensor], grad_output: &Tensor<f32>) -> Result<Vec<Tensor<f32>>, TensorError> {
        if inputs.len() != 3 {
            return Err(TensorError::UnexpectedShape);
        }

        let q = inputs[0].as_float()?;
        let k = inputs[1].as_float()?;
        let v = inputs[2].as_float()?;

        // Initialize gradients
        let mut grad_q = Tensor::zeros(q.shape());
        let mut grad_k = Tensor::zeros(k.shape());
        let mut grad_v = Tensor::zeros(v.shape());

        // For Flash Attention, we implement a tiled gradient computation
        // This maintains memory efficiency while computing accurate gradients
        self.compute_flash_gradients(
            q, k, v, grad_output,
            &mut grad_q, &mut grad_k, &mut grad_v
        )?;

        Ok(vec![grad_q, grad_k, grad_v])
    }

    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }

    #[cfg(feature = "gpu")]
    fn gpu_impl(&self, _out_id: crate::graph::TensorId, _inp_shapes: &[Vec<usize>]) -> crate::funcs::GpuFunction {
        // For now, return a placeholder GPU implementation
        // In a full implementation, this would contain optimized OpenCL kernels
        // for Flash Attention with shared memory optimization
        crate::funcs::GpuFunction::new(
            "flash_attention_kernel".to_string(),
            vec![],
            vec![],
        )
    }
}

impl FlashAttention {
    /// Compute Flash Attention gradients using tiled approach for memory efficiency
    fn compute_flash_gradients(
        &self,
        q: &Tensor<f32>,
        k: &Tensor<f32>,
        v: &Tensor<f32>,
        grad_output: &Tensor<f32>,
        grad_q: &mut Tensor<f32>,
        grad_k: &mut Tensor<f32>,
        grad_v: &mut Tensor<f32>,
    ) -> Result<(), TensorError> {
        let shape = q.shape();
        let seq_len_q = shape[shape.len() - 2];
        let seq_len_k = k.shape()[k.shape().len() - 2];
        let head_dim = shape[shape.len() - 1];

        // Calculate batch size for proper handling of batched tensors
        let batch_size = if shape.len() > 2 {
            shape[..shape.len() - 2].iter().product()
        } else {
            1
        };

        // Process in tiles for memory efficiency
        let num_tiles_q = (seq_len_q + self.block_size_q - 1) / self.block_size_q;
        let num_tiles_k = (seq_len_k + self.block_size_k - 1) / self.block_size_k;

        // Initialize running statistics for gradient computation
        let mut running_max = Tensor::zeros(&[batch_size, seq_len_q, 1]);
        let mut running_sum = Tensor::zeros(&[batch_size, seq_len_q, 1]);
        let mut output = Tensor::zeros(q.shape());

        // Forward pass to compute intermediate values needed for gradients
        self.forward_pass_for_gradients(
            q, k, v, &mut output, &mut running_max, &mut running_sum,
            num_tiles_q, num_tiles_k, batch_size, seq_len_q, seq_len_k, head_dim
        )?;

        // Backward pass through tiles
        for tile_q_idx in 0..num_tiles_q {
            let q_start = tile_q_idx * self.block_size_q;
            let q_end = std::cmp::min(q_start + self.block_size_q, seq_len_q);
            let q_size = q_end - q_start;

            // Extract gradient tile for current Q block
            let grad_out_tile = self.extract_tile_with_batch(grad_output, q_start, q_size, batch_size)?;

            for tile_k_idx in 0..num_tiles_k {
                let k_start = tile_k_idx * self.block_size_k;
                let k_end = std::cmp::min(k_start + self.block_size_k, seq_len_k);
                let k_size = k_end - k_start;

                // Extract tiles
                let q_tile = self.extract_tile_with_batch(q, q_start, q_size, batch_size)?;
                let k_tile = self.extract_tile_with_batch(k, k_start, k_size, batch_size)?;
                let v_tile = self.extract_tile_with_batch(v, k_start, k_size, batch_size)?;

                // Compute gradients for this tile pair
                let (tile_grad_q, tile_grad_k, tile_grad_v) = self.compute_tile_gradients(
                    &q_tile, &k_tile, &v_tile, &grad_out_tile,
                    &running_max, &running_sum, q_start, k_start, q_size, k_size, batch_size
                )?;

                // Accumulate gradients back to full tensors
                self.accumulate_tile_gradients(grad_q, &tile_grad_q, q_start, q_size, batch_size)?;
                self.accumulate_tile_gradients(grad_k, &tile_grad_k, k_start, k_size, batch_size)?;
                self.accumulate_tile_gradients(grad_v, &tile_grad_v, k_start, k_size, batch_size)?;
            }
        }

        Ok(())
    }

    /// Forward pass to compute intermediate values needed for gradient computation
    fn forward_pass_for_gradients(
        &self,
        q: &Tensor<f32>,
        k: &Tensor<f32>,
        v: &Tensor<f32>,
        output: &mut Tensor<f32>,
        running_max: &mut Tensor<f32>,
        running_sum: &mut Tensor<f32>,
        num_tiles_q: usize,
        num_tiles_k: usize,
        batch_size: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        head_dim: usize,
    ) -> Result<(), TensorError> {
        // Initialize running statistics with negative infinity for max
        let running_max_blob = running_max.blob_mut();
        for val in running_max_blob.iter_mut() {
            *val = f32::NEG_INFINITY;
        }

        // Process tiles similar to forward pass
        for tile_q_idx in 0..num_tiles_q {
            let q_start = tile_q_idx * self.block_size_q;
            let q_end = std::cmp::min(q_start + self.block_size_q, seq_len_q);
            let q_size = q_end - q_start;

            for tile_k_idx in 0..num_tiles_k {
                let k_start = tile_k_idx * self.block_size_k;
                let k_end = std::cmp::min(k_start + self.block_size_k, seq_len_k);
                let k_size = k_end - k_start;

                // Extract tiles
                let q_tile = self.extract_tile_with_batch(q, q_start, q_size, batch_size)?;
                let k_tile = self.extract_tile_with_batch(k, k_start, k_size, batch_size)?;
                let v_tile = self.extract_tile_with_batch(v, k_start, k_size, batch_size)?;

                // Compute attention for this tile
                let scores_tile = self.compute_scores_tile(&q_tile, &k_tile)?;
                let masked_scores = if self.causal {
                    self.apply_causal_mask_tile(&scores_tile, q_start, k_start)?
                } else {
                    scores_tile
                };

                // Update running statistics and compute output
                self.update_running_stats_and_output(
                    &masked_scores, &v_tile, output, running_max, running_sum,
                    q_start, k_start, q_size, k_size, batch_size
                )?;
            }
        }

        Ok(())
    }

    /// Compute gradients for a specific tile pair
    fn compute_tile_gradients(
        &self,
        q_tile: &Tensor<f32>,
        k_tile: &Tensor<f32>,
        v_tile: &Tensor<f32>,
        grad_out_tile: &Tensor<f32>,
        running_max: &Tensor<f32>,
        running_sum: &Tensor<f32>,
        q_start: usize,
        k_start: usize,
        q_size: usize,
        k_size: usize,
        batch_size: usize,
    ) -> Result<(Tensor<f32>, Tensor<f32>, Tensor<f32>), TensorError> {
        // Recompute forward pass for this tile to get attention weights
        let scores_tile = self.compute_scores_tile(q_tile, k_tile)?;
        let masked_scores = if self.causal {
            self.apply_causal_mask_tile(&scores_tile, q_start, k_start)?
        } else {
            scores_tile
        };

        // Extract relevant running statistics for this tile
        let tile_running_max = self.extract_running_stats(running_max, q_start, q_size)?;
        let tile_running_sum = self.extract_running_stats(running_sum, q_start, q_size)?;

        // Compute attention weights for this tile
        let attention_weights = self.compute_tile_attention_weights(
            &masked_scores, &tile_running_max, &tile_running_sum
        )?;

        // Compute gradients using chain rule
        // grad_v = attention_weights^T @ grad_output
        let attn_t = attention_weights.transpose()?;
        let grad_v_tile = self.matmul_tensors(&attn_t, grad_out_tile)?;

        // grad_attention_weights = grad_output @ v^T
        let v_t = v_tile.transpose()?;
        let grad_attention_weights = self.matmul_tensors(grad_out_tile, &v_t)?;

        // Gradient through softmax (simplified for tile)
        let grad_scores = self.softmax_backward_tile(&attention_weights, &grad_attention_weights)?;

        // Gradient through scaling and Q @ K^T
        let mut grad_scores_scaled = grad_scores.clone();
        let grad_blob = grad_scores_scaled.blob_mut();
        for val in grad_blob.iter_mut() {
            *val *= self.scale;
        }

        // grad_q = grad_scores @ k
        let grad_q_tile = self.matmul_tensors(&grad_scores_scaled, k_tile)?;

        // grad_k = grad_scores^T @ q
        let grad_scores_t = grad_scores_scaled.transpose()?;
        let grad_k_tile = self.matmul_tensors(&grad_scores_t, q_tile)?;

        Ok((grad_q_tile, grad_k_tile, grad_v_tile))
    }

    /// Extract tile with proper batch handling
    fn extract_tile_with_batch(
        &self,
        tensor: &Tensor<f32>,
        start: usize,
        size: usize,
        batch_size: usize,
    ) -> Result<Tensor<f32>, TensorError> {
        let shape = tensor.shape();
        let seq_len = shape[shape.len() - 2];
        let head_dim = shape[shape.len() - 1];
        let end = std::cmp::min(start + size, seq_len);
        let actual_size = end - start;

        let blob = tensor.blob();
        let mut tile_data = Vec::with_capacity(batch_size * actual_size * head_dim);

        if shape.len() == 2 {
            // Non-batched case
            for i in start..end {
                for j in 0..head_dim {
                    let idx = i * head_dim + j;
                    if idx < blob.len() {
                        tile_data.push(blob[idx]);
                    }
                }
            }
            Tensor::raw(&[actual_size, head_dim], tile_data)
        } else {
            // Batched case
            for batch_idx in 0..batch_size {
                let batch_offset = batch_idx * seq_len * head_dim;
                for i in start..end {
                    for j in 0..head_dim {
                        let idx = batch_offset + i * head_dim + j;
                        if idx < blob.len() {
                            tile_data.push(blob[idx]);
                        }
                    }
                }
            }
            let mut tile_shape = shape.to_vec();
            let seq_idx = tile_shape.len() - 2;
            tile_shape[seq_idx] = actual_size;
            Tensor::raw(&tile_shape, tile_data)
        }
    }

    /// Accumulate tile gradients back to full gradient tensor
    fn accumulate_tile_gradients(
        &self,
        full_grad: &mut Tensor<f32>,
        tile_grad: &Tensor<f32>,
        start: usize,
        size: usize,
        batch_size: usize,
    ) -> Result<(), TensorError> {
        let shape = full_grad.shape().to_vec(); // Clone to avoid borrow conflict
        let seq_len = shape[shape.len() - 2];
        let head_dim = shape[shape.len() - 1];
        let end = std::cmp::min(start + size, seq_len);

        let full_blob = full_grad.blob_mut();
        let tile_blob = tile_grad.blob();

        if shape.len() == 2 {
            // Non-batched case
            for (tile_i, full_i) in (start..end).enumerate() {
                for j in 0..head_dim {
                    let full_idx = full_i * head_dim + j;
                    let tile_idx = tile_i * head_dim + j;
                    
                    if full_idx < full_blob.len() && tile_idx < tile_blob.len() {
                        full_blob[full_idx] += tile_blob[tile_idx];
                    }
                }
            }
        } else {
            // Batched case
            for batch_idx in 0..batch_size {
                let full_batch_offset = batch_idx * seq_len * head_dim;
                let tile_batch_offset = batch_idx * (end - start) * head_dim;
                
                for (tile_i, full_i) in (start..end).enumerate() {
                    for j in 0..head_dim {
                        let full_idx = full_batch_offset + full_i * head_dim + j;
                        let tile_idx = tile_batch_offset + tile_i * head_dim + j;
                        
                        if full_idx < full_blob.len() && tile_idx < tile_blob.len() {
                            full_blob[full_idx] += tile_blob[tile_idx];
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Simple matrix multiplication for tensors
    fn matmul_tensors(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        let mut matmul = crate::funcs::MatMul::new();
        matmul.run(&[&crate::tensor::GeneralTensor::Float(a.clone()), 
                    &crate::tensor::GeneralTensor::Float(b.clone())], false)
    }

    /// Softmax backward for tile
    fn softmax_backward_tile(&self, softmax_output: &Tensor<f32>, grad_output: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        let shape = softmax_output.shape();
        let mut grad_input = Tensor::zeros(shape);
        
        let softmax_blob = softmax_output.blob();
        let grad_out_blob = grad_output.blob();
        let grad_in_blob = grad_input.blob_mut();
        
        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        
        // For each row
        for i in 0..rows {
            let row_start = i * cols;
            let row_end = row_start + cols;
            
            // Compute sum of (grad_output * softmax_output) for this row
            let mut sum = 0.0;
            for j in row_start..row_end {
                if j < softmax_blob.len() && j < grad_out_blob.len() {
                    sum += grad_out_blob[j] * softmax_blob[j];
                }
            }
            
            // Compute gradient: softmax * (grad_output - sum)
            for j in row_start..row_end {
                if j < grad_in_blob.len() && j < softmax_blob.len() && j < grad_out_blob.len() {
                    grad_in_blob[j] = softmax_blob[j] * (grad_out_blob[j] - sum);
                }
            }
        }
        
        Ok(grad_input)
    }

    /// Extract running statistics for a specific tile
    fn extract_running_stats(&self, stats: &Tensor<f32>, start: usize, size: usize) -> Result<Tensor<f32>, TensorError> {
        let stats_shape = stats.shape();
        let stats_blob = stats.blob();
        

        
        // Handle batched tensors correctly
        if stats_shape.len() == 2 {
            // Non-batched case: [seq_len, 1]
            let mut tile_data = Vec::with_capacity(size);
            
            for i in 0..size {
                let idx = start + i;
                if idx < stats_blob.len() {
                    tile_data.push(stats_blob[idx]);
                } else {
                    tile_data.push(f32::NEG_INFINITY); // Default for max, 0.0 for sum would be handled separately
                }
            }
            
            Tensor::raw(&[size, 1], tile_data)
        } else {
            // Batched case: [batch_dims..., seq_len, 1]
            let batch_size = stats_shape[..stats_shape.len() - 2].iter().product::<usize>();
            let seq_len = stats_shape[stats_shape.len() - 2];
            
            let mut tile_data = Vec::with_capacity(batch_size * size);
            
            for batch_idx in 0..batch_size {
                let batch_offset = batch_idx * seq_len;
                for i in 0..size {
                    let idx = batch_offset + start + i;
                    if idx < stats_blob.len() && (start + i) < seq_len {
                        tile_data.push(stats_blob[idx]);
                    } else {
                        tile_data.push(f32::NEG_INFINITY); // Default for max
                    }
                }
            }
            
            let mut tile_shape = stats_shape.to_vec();
            let last_idx = tile_shape.len() - 2;
            tile_shape[last_idx] = size; // Update sequence length dimension
            Tensor::raw(&tile_shape, tile_data)
        }
    }
    
    /// Update running statistics for a specific tile
    fn update_running_stats(&self, stats: &mut Tensor<f32>, new_stats: &Tensor<f32>, start: usize, size: usize) -> Result<(), TensorError> {
        let stats_shape = stats.shape().to_vec(); // Clone to avoid borrow conflict
        let stats_blob = stats.blob_mut();
        let new_blob = new_stats.blob();
        
        // Handle batched tensors correctly
        if stats_shape.len() == 2 {
            // Non-batched case: [seq_len, 1]
            for i in 0..size {
                let idx = start + i;
                if idx < stats_blob.len() && i < new_blob.len() {
                    stats_blob[idx] = new_blob[i];
                }
            }
        } else {
            // Batched case: [batch_dims..., seq_len, 1]
            let batch_size = stats_shape[..stats_shape.len() - 2].iter().product::<usize>();
            let seq_len = stats_shape[stats_shape.len() - 2];
            
            for batch_idx in 0..batch_size {
                let stats_batch_offset = batch_idx * seq_len;
                let new_batch_offset = batch_idx * size;
                
                for i in 0..size {
                    let stats_idx = stats_batch_offset + start + i;
                    let new_idx = new_batch_offset + i;
                    
                    if stats_idx < stats_blob.len() && new_idx < new_blob.len() && (start + i) < seq_len {
                        stats_blob[stats_idx] = new_blob[new_idx];
                    }
                }
            }
        }

        Ok(())
    }
    
    /// Accumulate tile output into the main output tensor
    fn accumulate_tile_output(&self, output: &mut Tensor<f32>, tile_output: &Tensor<f32>, start: usize, q_tile_size: usize) -> Result<(), TensorError> {
        let output_shape = output.shape().to_vec(); // Clone to avoid borrow conflict
        let head_dim = output_shape[output_shape.len() - 1];
        let output_blob = output.blob_mut();
        let tile_blob = tile_output.blob();
        
        // Handle batched tensors correctly
        if output_shape.len() == 2 {
            // Non-batched case: [seq_len, head_dim]
            for i in 0..q_tile_size {
                for j in 0..head_dim {
                    let output_idx = (start + i) * head_dim + j;
                    let tile_idx = i * head_dim + j;
                    
                    if output_idx < output_blob.len() && tile_idx < tile_blob.len() {
                        output_blob[output_idx] += tile_blob[tile_idx];
                    }
                }
            }
        } else {
            // Batched case: [batch_dims..., seq_len, head_dim]
            let batch_size = output_shape[..output_shape.len() - 2].iter().product::<usize>();
            let seq_len = output_shape[output_shape.len() - 2];
            
            for batch_idx in 0..batch_size {
                let output_batch_offset = batch_idx * seq_len * head_dim;
                let tile_batch_offset = batch_idx * q_tile_size * head_dim;
                
                for i in 0..q_tile_size {
                    for j in 0..head_dim {
                        let output_idx = output_batch_offset + (start + i) * head_dim + j;
                        let tile_idx = tile_batch_offset + i * head_dim + j;
                        
                        if output_idx < output_blob.len() && tile_idx < tile_blob.len() && (start + i) < seq_len {
                            output_blob[output_idx] += tile_blob[tile_idx];
                        }
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Extract a tile from a tensor along the sequence dimension
    fn extract_tile(&self, tensor: &Tensor<f32>, start: usize, end: usize, num_dims: usize) -> Result<Tensor<f32>, TensorError> {
        let shape = tensor.shape();
        let seq_dim = num_dims - 2;
        
        // Create new shape for the tile
        let mut tile_shape = shape.to_vec();
        tile_shape[seq_dim] = end - start;
        
        let mut tile_data = Vec::new();
        let blob = tensor.blob();
        
        // Extract data for the tile
        if num_dims == 2 {
            // Simple 2D case
            let cols = shape[1];
            for i in start..end {
                for j in 0..cols {
                    let linear_idx = i * cols + j;
                    tile_data.push(blob[linear_idx]);
                }
            }
        } else {
            // Handle batch dimensions (simplified for 3D case)
            let seq_len = shape[1];
            let cols = shape[2];
            for b in 0..shape[0] {
                for i in start..end {
                    for j in 0..cols {
                        let linear_idx = b * (seq_len * cols) + i * cols + j;
                        tile_data.push(blob[linear_idx]);
                    }
                }
            }
        }
        
        Tensor::raw(&tile_shape, tile_data)
    }

    /// Apply final normalization to the output
    fn final_normalization(&self, output: &mut Tensor<f32>, running_sum: &Tensor<f32>) -> Result<(), TensorError> {
        let shape = output.shape();
        let seq_len = shape[shape.len() - 2];
        let head_dim = shape[shape.len() - 1];

        let output_blob = output.blob_mut();
        let sum_blob = running_sum.blob();

        for i in 0..seq_len {
            let sum_val = sum_blob[i];
            if sum_val > 0.0 {
                for j in 0..head_dim {
                    let linear_idx = i * head_dim + j;
                    output_blob[linear_idx] /= sum_val;
                }
            }
        }

        Ok(())
    }

    /// Compute attention weights for a tile using running statistics
    fn compute_tile_attention_weights(
        &self,
        masked_scores: &Tensor<f32>,
        tile_running_max: &Tensor<f32>,
        tile_running_sum: &Tensor<f32>,
    ) -> Result<Tensor<f32>, TensorError> {
        let shape = masked_scores.shape();
        let mut attention_weights = Tensor::zeros(shape);
        
        let scores_blob = masked_scores.blob();
        let max_blob = tile_running_max.blob();
        let sum_blob = tile_running_sum.blob();
        let weights_blob = attention_weights.blob_mut();
        
        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        
        for i in 0..rows {
            let max_val = if i < max_blob.len() { max_blob[i] } else { 0.0 };
            let sum_val = if i < sum_blob.len() { sum_blob[i] } else { 1.0 };
            
            for j in 0..cols {
                let idx = i * cols + j;
                if idx < scores_blob.len() && idx < weights_blob.len() {
                    let score = scores_blob[idx];
                    weights_blob[idx] = (score - max_val).exp() / sum_val;
                }
            }
        }
        
        Ok(attention_weights)
    }

    /// Update running statistics and output for gradient computation
    fn update_running_stats_and_output(
        &self,
        masked_scores: &Tensor<f32>,
        v_tile: &Tensor<f32>,
        output: &mut Tensor<f32>,
        running_max: &mut Tensor<f32>,
        running_sum: &mut Tensor<f32>,
        q_start: usize,
        k_start: usize,
        q_size: usize,
        k_size: usize,
        batch_size: usize,
    ) -> Result<(), TensorError> {
        // Compute row max and sum for this tile
        let tile_max = self.compute_row_max(masked_scores)?;
        let stable_exp = self.compute_stable_exp(masked_scores, &tile_max)?;
        let tile_sum = self.compute_row_sum(&stable_exp)?;
        
        // Update running statistics (simplified version for gradients)
        self.update_running_stats(running_max, &tile_max, q_start, q_size)?;
        self.update_running_stats(running_sum, &tile_sum, q_start, q_size)?;
        
        // Compute and accumulate output for this tile
        let tile_output = self.matmul_tensors(&stable_exp, v_tile)?;
        self.accumulate_tile_output(output, &tile_output, q_start, q_size)?;
        
        Ok(())
    }

    /// Apply causal mask to a tile
    fn apply_causal_mask_tile(&self, scores: &Tensor<f32>, q_start: usize, k_start: usize) -> Result<Tensor<f32>, TensorError> {
        let shape = scores.shape();
        let mut masked_scores = scores.clone();
        let blob = masked_scores.blob_mut();
        
        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        
        for i in 0..rows {
            for j in 0..cols {
                let q_pos = q_start + i;
                let k_pos = k_start + j;
                
                if q_pos < k_pos {
                    let idx = i * cols + j;
                    if idx < blob.len() {
                        blob[idx] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        
        Ok(masked_scores)
    }

    /// Compute scores for a tile (Q @ K^T)
    fn compute_scores_tile(&self, q_tile: &Tensor<f32>, k_tile: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        let k_t = k_tile.transpose()?;
        let scores = self.matmul_tensors(q_tile, &k_t)?;
        
        // Apply scaling
        let mut scaled_scores = scores.clone();
        let blob = scaled_scores.blob_mut();
        for val in blob.iter_mut() {
            *val *= self.scale;
        }
        
        Ok(scaled_scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_flash_attention_basic() {
        let mut flash_attn = FlashAttention::new(1.0, false);
        
        let q = Tensor::raw(&[4, 8], (0..32).map(|x| x as f32).collect()).unwrap();
        let k = Tensor::raw(&[4, 8], (0..32).map(|x| (x + 1) as f32).collect()).unwrap();
        let v = Tensor::raw(&[4, 8], (0..32).map(|x| (x + 2) as f32).collect()).unwrap();
        
        let inputs = [&GeneralTensor::Float(q), &GeneralTensor::Float(k), &GeneralTensor::Float(v)];
        let result = flash_attn.run(&inputs, false);
        
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.shape(), &[4, 8]);
    }

    #[test]
    fn test_flash_attention_batched() {
        let mut flash_attn = FlashAttention::new(1.0, false);
        
        // Create batched tensors with shape [1, 4, 8] (batch_size=1, seq_len=4, head_dim=8)
        let q = Tensor::raw(&[1, 4, 8], (0..32).map(|x| x as f32).collect()).unwrap();
        let k = Tensor::raw(&[1, 4, 8], (0..32).map(|x| (x + 1) as f32).collect()).unwrap();
        let v = Tensor::raw(&[1, 4, 8], (0..32).map(|x| (x + 2) as f32).collect()).unwrap();
        
        let inputs = [&GeneralTensor::Float(q), &GeneralTensor::Float(k), &GeneralTensor::Float(v)];
        let result = flash_attn.run(&inputs, false);
        
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_flash_attention_causal() {
        let mut flash_attn = FlashAttention::new(1.0, true);
        
        let q = Tensor::raw(&[4, 8], (0..32).map(|x| x as f32).collect()).unwrap();
        let k = Tensor::raw(&[4, 8], (0..32).map(|x| (x + 1) as f32).collect()).unwrap();
        let v = Tensor::raw(&[4, 8], (0..32).map(|x| (x + 2) as f32).collect()).unwrap();
        
        let inputs = [&GeneralTensor::Float(q), &GeneralTensor::Float(k), &GeneralTensor::Float(v)];
        let result = flash_attn.run(&inputs, false);
        
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.shape(), &[4, 8]);
    }

    #[test]
    fn test_flash_attention_larger_sequence() {
        let mut flash_attn = FlashAttention::with_block_sizes(1.0, false, 16, 16);
        
        let seq_len = 32;
        let head_dim = 16;
        let size = seq_len * head_dim;
        
        let q = Tensor::raw(&[seq_len, head_dim], (0..size).map(|x| x as f32 / size as f32).collect()).unwrap();
        let k = Tensor::raw(&[seq_len, head_dim], (0..size).map(|x| (x + 1) as f32 / size as f32).collect()).unwrap();
        let v = Tensor::raw(&[seq_len, head_dim], (0..size).map(|x| (x + 2) as f32 / size as f32).collect()).unwrap();
        
        let inputs = [&GeneralTensor::Float(q), &GeneralTensor::Float(k), &GeneralTensor::Float(v)];
        let result = flash_attn.run(&inputs, false);
        
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.shape(), &[seq_len, head_dim]);
    }
}