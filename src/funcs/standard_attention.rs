use crate::funcs::{Function, MatMul, Coeff, Softmax, TrilMask, Dropout};
use crate::tensor::{GeneralTensor, Tensor, TensorError, TensorOps, TensorMutOps};

/// Standard Multi-Head Attention implementation
/// 
/// This implements the traditional attention mechanism as described in 
/// "Attention Is All You Need" with O(N²) memory complexity.
/// 
/// The implementation matches the current inline attention in GPT model:
/// 1. Linear projections for Q, K, V
/// 2. Scaled dot-product attention: softmax(QK^T / sqrt(d_k))V
/// 3. Optional causal masking for autoregressive models
/// 4. Optional dropout for regularization
#[derive(Clone, Debug)]
pub struct StandardAttention {
    /// Number of attention heads
    num_heads: usize,
    /// Dimension of each attention head
    head_dim: usize,
    /// Whether to apply causal masking
    causal: bool,
    /// Dropout probability
    dropout: f32,
    /// Scaling factor (typically 1/sqrt(head_dim))
    scale: f32,
}

impl StandardAttention {
    /// Create new Standard Attention
    pub fn new(num_heads: usize, head_dim: usize, causal: bool, dropout: f32) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self {
            num_heads,
            head_dim,
            causal,
            dropout,
            scale,
        }
    }

    /// Create Standard Attention with custom scale
    pub fn with_scale(num_heads: usize, head_dim: usize, causal: bool, dropout: f32, scale: f32) -> Self {
        Self {
            num_heads,
            head_dim,
            causal,
            dropout,
            scale,
        }
    }

    /// Compute single-head attention
    fn compute_single_head_attention(
        &self,
        q: &Tensor<f32>,
        k: &Tensor<f32>,
        v: &Tensor<f32>,
        seq_len: usize,
    ) -> Result<Tensor<f32>, TensorError> {
        // Q @ K^T
        let mut matmul1 = MatMul::new();
        let q_t = q.transpose()?;
        let qk = matmul1.run(&[&GeneralTensor::Float(k.clone()), &GeneralTensor::Float(q_t)], false)?;

        // Scale by 1/sqrt(head_dim)
        let mut coeff = Coeff::new(self.scale);
        let scaled_qk = coeff.run(&[&GeneralTensor::Float(qk)], false)?;

        // Apply causal masking if enabled
        let masked_qk = if self.causal {
            let mut tril_mask = TrilMask::new(seq_len);
            tril_mask.run(&[&GeneralTensor::Float(scaled_qk)], false)?
        } else {
            scaled_qk
        };

        // Softmax
        let mut softmax = Softmax::new();
        let attention_weights = softmax.run(&[&GeneralTensor::Float(masked_qk)], false)?;

        // Apply dropout
        let dropped_weights = if self.dropout > 0.0 {
            let mut dropout = Dropout::new(self.dropout);
            dropout.run(&[&GeneralTensor::Float(attention_weights)], true)?
        } else {
            attention_weights
        };

        // Apply attention to values: attention_weights @ V
        let mut matmul2 = MatMul::new();
        matmul2.run(&[&GeneralTensor::Float(dropped_weights), &GeneralTensor::Float(v.clone())], false)
    }
}

impl Function for StandardAttention {
    fn run(&mut self, inputs: &[&GeneralTensor], training: bool) -> Result<Tensor<f32>, TensorError> {
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

        let seq_len = q_shape[q_shape.len() - 2];
        let embed_dim = q_shape[q_shape.len() - 1];

        // Validate that embed_dim is divisible by num_heads
        if embed_dim % self.num_heads != 0 {
            return Err(TensorError::UnexpectedShape);
        }

        let actual_head_dim = embed_dim / self.num_heads;
        if actual_head_dim != self.head_dim {
            return Err(TensorError::UnexpectedShape);
        }

        // For simplicity, this implementation assumes inputs are already projected Q, K, V
        // In a full implementation, we would include the linear projections here
        
        // Split into multiple heads and compute attention for each
        let mut head_outputs = Vec::with_capacity(self.num_heads);
        
        for head_idx in 0..self.num_heads {
            let head_start = head_idx * self.head_dim;
            let head_end = head_start + self.head_dim;
            
            // Extract head-specific Q, K, V (simplified - assumes already split)
            let q_head = self.extract_head(q, head_start, head_end)?;
            let k_head = self.extract_head(k, head_start, head_end)?;
            let v_head = self.extract_head(v, head_start, head_end)?;
            
            let head_output = self.compute_single_head_attention(&q_head, &k_head, &v_head, seq_len)?;
            head_outputs.push(head_output);
        }

        // Concatenate head outputs
        self.concat_heads(&head_outputs)
    }

    fn grad(&self, inputs: &[&GeneralTensor], grad_output: &Tensor<f32>) -> Result<Vec<Tensor<f32>>, TensorError> {
        if inputs.len() != 3 {
            return Err(TensorError::UnexpectedShape);
        }

        let q = inputs[0].as_float()?;
        let k = inputs[1].as_float()?;
        let v = inputs[2].as_float()?;

        // Validate input dimensions
        let q_shape = q.shape();
        let seq_len = q_shape[q_shape.len() - 2];
        let embed_dim = q_shape[q_shape.len() - 1];
        let actual_head_dim = embed_dim / self.num_heads;

        // Initialize gradients
        let mut grad_q = Tensor::zeros(q.shape());
        let mut grad_k = Tensor::zeros(k.shape());
        let mut grad_v = Tensor::zeros(v.shape());

        // Compute gradients for each head
        for head_idx in 0..self.num_heads {
            let head_start = head_idx * self.head_dim;
            let head_end = head_start + self.head_dim;
            
            // Extract head-specific tensors
            let q_head = self.extract_head(q, head_start, head_end)?;
            let k_head = self.extract_head(k, head_start, head_end)?;
            let v_head = self.extract_head(v, head_start, head_end)?;
            
            // Extract head-specific gradient
            let grad_out_head = self.extract_head_grad(grad_output, head_idx, actual_head_dim)?;
            
            // Compute head gradients
            let (head_grad_q, head_grad_k, head_grad_v) = self.compute_head_gradients(
                &q_head, &k_head, &v_head, &grad_out_head, seq_len
            )?;
            
            // Accumulate gradients back to full tensors
            self.accumulate_head_gradients(&mut grad_q, &head_grad_q, head_start, head_end)?;
            self.accumulate_head_gradients(&mut grad_k, &head_grad_k, head_start, head_end)?;
            self.accumulate_head_gradients(&mut grad_v, &head_grad_v, head_start, head_end)?;
        }

        Ok(vec![grad_q, grad_k, grad_v])
    }

    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }

    #[cfg(feature = "gpu")]
    fn gpu_impl(&self, _out_id: crate::graph::TensorId, _inp_shapes: &[Vec<usize>]) -> crate::funcs::GpuFunction {
        // Placeholder GPU implementation
        crate::funcs::GpuFunction::new(
            "standard_attention_kernel".to_string(),
            vec![],
            vec![],
        )
    }
}

impl StandardAttention {
    /// Extract a specific head from the input tensor
    fn extract_head(&self, tensor: &Tensor<f32>, start: usize, end: usize) -> Result<Tensor<f32>, TensorError> {
        let shape = tensor.shape();
        let seq_len = shape[shape.len() - 2];
        let embed_dim = shape[shape.len() - 1];
        
        let mut head_data = Vec::with_capacity(seq_len * self.head_dim);
        let blob = tensor.blob();
        
        for i in 0..seq_len {
            for j in start..end {
                let idx = i * embed_dim + j;
                if idx < blob.len() {
                    head_data.push(blob[idx]);
                }
            }
        }
        
        Tensor::raw(&[seq_len, self.head_dim], head_data)
    }

    /// Concatenate outputs from multiple heads
    fn concat_heads(&self, head_outputs: &[Tensor<f32>]) -> Result<Tensor<f32>, TensorError> {
        if head_outputs.is_empty() {
            return Err(TensorError::UnexpectedShape);
        }

        let head_shape = head_outputs[0].shape();
        let seq_len = head_shape[0];
        let total_dim = self.num_heads * self.head_dim;
        
        let mut concat_data = Vec::with_capacity(seq_len * total_dim);
        
        for i in 0..seq_len {
            for head in head_outputs {
                let head_blob = head.blob();
                for j in 0..self.head_dim {
                    let idx = i * self.head_dim + j;
                    if idx < head_blob.len() {
                        concat_data.push(head_blob[idx]);
                    }
                }
            }
        }
        
        Tensor::raw(&[seq_len, total_dim], concat_data)
    }

    /// Extract head-specific gradient from output gradient
    fn extract_head_grad(&self, grad_output: &Tensor<f32>, head_idx: usize, head_dim: usize) -> Result<Tensor<f32>, TensorError> {
        let shape = grad_output.shape();
        let seq_len = shape[shape.len() - 2];
        let embed_dim = shape[shape.len() - 1];
        
        let head_start = head_idx * head_dim;
        let head_end = head_start + head_dim;
        
        let mut head_grad_data = Vec::with_capacity(seq_len * head_dim);
        let blob = grad_output.blob();
        
        for i in 0..seq_len {
            for j in head_start..head_end {
                let idx = i * embed_dim + j;
                if idx < blob.len() {
                    head_grad_data.push(blob[idx]);
                }
            }
        }
        
        let mut head_shape = shape.to_vec();
        let last_idx = head_shape.len() - 1;
        head_shape[last_idx] = head_dim;
        
        Tensor::raw(&head_shape, head_grad_data)
    }

    /// Compute gradients for a single attention head using chain rule
    fn compute_head_gradients(
        &self,
        q_head: &Tensor<f32>,
        k_head: &Tensor<f32>,
        v_head: &Tensor<f32>,
        grad_output: &Tensor<f32>,
        seq_len: usize,
    ) -> Result<(Tensor<f32>, Tensor<f32>, Tensor<f32>), TensorError> {
        // Forward pass to get intermediate values needed for gradients
        let mut matmul1 = MatMul::new();
        let q_t = q_head.transpose()?;
        let qk = matmul1.run(&[&GeneralTensor::Float(k_head.clone()), &GeneralTensor::Float(q_t)], false)?;

        let mut coeff = Coeff::new(self.scale);
        let scaled_qk = coeff.run(&[&GeneralTensor::Float(qk)], false)?;

        let masked_qk = if self.causal {
            let mut tril_mask = TrilMask::new(seq_len);
            tril_mask.run(&[&GeneralTensor::Float(scaled_qk)], false)?
        } else {
            scaled_qk
        };

        let mut softmax = Softmax::new();
        let attention_weights = softmax.run(&[&GeneralTensor::Float(masked_qk)], false)?;

        // Gradient computation using chain rule
        // grad_v = attention_weights^T @ grad_output
        let mut grad_v_matmul = MatMul::new();
        let attn_t = attention_weights.transpose()?;
        let grad_v = grad_v_matmul.run(&[&GeneralTensor::Float(attn_t), &GeneralTensor::Float(grad_output.clone())], false)?;

        // grad_attention_weights = grad_output @ v^T
        let mut grad_attn_matmul = MatMul::new();
        let v_t = v_head.transpose()?;
        let grad_attention_weights = grad_attn_matmul.run(&[&GeneralTensor::Float(grad_output.clone()), &GeneralTensor::Float(v_t)], false)?;

        // Gradient through softmax
        let grad_masked_qk = self.softmax_backward(&attention_weights, &grad_attention_weights)?;

        // Gradient through scaling
        let mut grad_scaled_qk = grad_masked_qk.clone();
        let grad_blob = grad_scaled_qk.blob_mut();
        for val in grad_blob.iter_mut() {
            *val *= self.scale;
        }

        // Gradient through Q @ K^T
        // grad_q = grad_scaled_qk @ k
        let mut grad_q_matmul = MatMul::new();
        let grad_q = grad_q_matmul.run(&[&GeneralTensor::Float(grad_scaled_qk.clone()), &GeneralTensor::Float(k_head.clone())], false)?;

        // grad_k = grad_scaled_qk^T @ q
        let mut grad_k_matmul = MatMul::new();
        let grad_scaled_qk_t = grad_scaled_qk.transpose()?;
        let grad_k = grad_k_matmul.run(&[&GeneralTensor::Float(grad_scaled_qk_t), &GeneralTensor::Float(q_head.clone())], false)?;

        Ok((grad_q, grad_k, grad_v))
    }

    /// Backward pass through softmax
    fn softmax_backward(&self, softmax_output: &Tensor<f32>, grad_output: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        let shape = softmax_output.shape();
        let seq_len = shape[shape.len() - 2];
        let mut grad_input = Tensor::zeros(shape);
        
        let softmax_blob = softmax_output.blob();
        let grad_out_blob = grad_output.blob();
        let grad_in_blob = grad_input.blob_mut();
        
        // For each row (query position)
        for i in 0..seq_len {
            let row_start = i * seq_len;
            let row_end = row_start + seq_len;
            
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

    /// Accumulate head gradients back to full tensor
    fn accumulate_head_gradients(
        &self,
        full_grad: &mut Tensor<f32>,
        head_grad: &Tensor<f32>,
        head_start: usize,
        head_end: usize,
    ) -> Result<(), TensorError> {
        let shape = full_grad.shape();
        let seq_len = shape[shape.len() - 2];
        let embed_dim = shape[shape.len() - 1];
        
        let full_blob = full_grad.blob_mut();
        let head_blob = head_grad.blob();
        
        for i in 0..seq_len {
            for (j, head_j) in (head_start..head_end).enumerate() {
                let full_idx = i * embed_dim + head_j;
                let head_idx = i * self.head_dim + j;
                
                if full_idx < full_blob.len() && head_idx < head_blob.len() {
                    full_blob[full_idx] += head_blob[head_idx];
                }
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_attention_basic() {
        let seq_len = 4;
        let head_dim = 2;
        let num_heads = 1;
        
        // Create test inputs (Q, K, V)
        let q_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let k_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let v_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let q = Tensor::raw(&[seq_len, head_dim], q_data).unwrap();
        let k = Tensor::raw(&[seq_len, head_dim], k_data).unwrap();
        let v = Tensor::raw(&[seq_len, head_dim], v_data).unwrap();

        let mut std_attn = StandardAttention::new(num_heads, head_dim, false, 0.0);
        let q_gen = GeneralTensor::Float(q);
        let k_gen = GeneralTensor::Float(k);
        let v_gen = GeneralTensor::Float(v);
        let inputs = vec![&q_gen, &k_gen, &v_gen];

        let result = std_attn.run(&inputs, false);
        assert!(result.is_ok(), "Standard attention should execute successfully");
        
        let output = result.unwrap();
        assert_eq!(output.shape(), &[seq_len, head_dim]);
    }

    #[test]
    fn test_standard_attention_causal() {
        let seq_len = 4;
        let head_dim = 2;
        let num_heads = 1;
        
        let q_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let k_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let v_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let q = Tensor::raw(&[seq_len, head_dim], q_data).unwrap();
        let k = Tensor::raw(&[seq_len, head_dim], k_data).unwrap();
        let v = Tensor::raw(&[seq_len, head_dim], v_data).unwrap();

        let mut std_attn = StandardAttention::new(num_heads, head_dim, true, 0.0);
        let q_gen = GeneralTensor::Float(q);
        let k_gen = GeneralTensor::Float(k);
        let v_gen = GeneralTensor::Float(v);
        let inputs = vec![&q_gen, &k_gen, &v_gen];

        let result = std_attn.run(&inputs, false);
        assert!(result.is_ok(), "Causal standard attention should execute successfully");
        
        let output = result.unwrap();
        assert_eq!(output.shape(), &[seq_len, head_dim]);
    }
}