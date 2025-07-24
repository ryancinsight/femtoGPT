use crate::funcs::{Function, MatMul, Coeff, Softmax, TrilMask, Dropout};
use crate::tensor::{GeneralTensor, Tensor, TensorError, TensorOps};

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

    fn grad(&self, inputs: &[&GeneralTensor], _grad_output: &Tensor<f32>) -> Result<Vec<Tensor<f32>>, TensorError> {
        if inputs.len() != 3 {
            return Err(TensorError::UnexpectedShape);
        }

        let q = inputs[0].as_float()?;
        let k = inputs[1].as_float()?;
        let v = inputs[2].as_float()?;

        // Placeholder gradients - in practice, these would be computed using
        // the chain rule through all the attention operations
        let grad_q = Tensor::zeros(q.shape());
        let grad_k = Tensor::zeros(k.shape());
        let grad_v = Tensor::zeros(v.shape());

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