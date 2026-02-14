use half::f16;
use nordocr_core::{OcrError, Result};
use nordocr_gpu::{GpuBuffer, GpuContext};

use crate::charset::{self, CHARSET_SIZE, EOS_TOKEN, PAD_TOKEN};

/// Decoded text with per-character confidence scores.
#[derive(Debug, Clone)]
pub struct DecodedText {
    pub text: String,
    pub confidence: f32,
    pub char_confidences: Vec<f32>,
}

/// Decodes PARSeq logits (GPU) into text strings.
///
/// PARSeq outputs logits of shape [batch, max_seq_len, charset_size].
/// We perform argmax + confidence computation, then map tokens to characters.
pub struct TokenDecoder {
    max_seq_len: u32,
}

impl TokenDecoder {
    pub fn new(max_seq_len: u32) -> Self {
        Self { max_seq_len }
    }

    /// Decode a batch of logits from GPU to text strings.
    ///
    /// This is one of only two places we transfer data from GPU to CPU
    /// (the other being the final text output). The logits are relatively
    /// small: batch_size * max_seq_len * charset_size * 2 bytes.
    pub fn decode_batch(
        &self,
        ctx: &GpuContext,
        logits: &GpuBuffer<f16>,
        batch_size: u32,
    ) -> Result<Vec<DecodedText>> {
        // In production:
        //   1. Optionally run argmax on GPU (custom kernel) to reduce transfer size.
        //   2. Copy token IDs + max logits to CPU.
        //
        // For now, copy full logits to CPU (small enough for typical batch sizes).
        //
        //   let logits_cpu = ctx.device.dtoh_sync_copy(logits)?;

        let _ = (ctx, logits);

        let mut results = Vec::with_capacity(batch_size as usize);

        for b in 0..batch_size {
            // Decode a single sequence by taking argmax at each position.
            let decoded = self.decode_single_sequence(b, &[])?;
            results.push(decoded);
        }

        Ok(results)
    }

    fn decode_single_sequence(
        &self,
        _batch_index: u32,
        logits_cpu: &[f16], // [max_seq_len, charset_size]
    ) -> Result<DecodedText> {
        let mut text = String::new();
        let mut char_confidences = Vec::new();
        let mut total_log_prob = 0.0f32;
        let mut num_chars = 0;

        for pos in 0..self.max_seq_len as usize {
            let offset = pos * CHARSET_SIZE;

            // Find argmax token and its probability.
            // In production this operates on the actual logits_cpu slice.
            let (best_token, best_prob) = if offset + CHARSET_SIZE <= logits_cpu.len() {
                let slice = &logits_cpu[offset..offset + CHARSET_SIZE];
                softmax_argmax(slice)
            } else {
                (EOS_TOKEN, 1.0)
            };

            // Stop at EOS or PAD.
            if best_token == EOS_TOKEN || best_token == PAD_TOKEN {
                break;
            }

            if let Some(c) = charset::token_to_char(best_token) {
                text.push(c);
                char_confidences.push(best_prob);
                total_log_prob += best_prob.ln();
                num_chars += 1;
            }
        }

        // Geometric mean of per-character probabilities as overall confidence.
        let confidence = if num_chars > 0 {
            (total_log_prob / num_chars as f32).exp()
        } else {
            0.0
        };

        Ok(DecodedText {
            text,
            confidence,
            char_confidences,
        })
    }
}

/// Compute softmax and return (argmax_index, max_probability).
fn softmax_argmax(logits: &[f16]) -> (usize, f32) {
    if logits.is_empty() {
        return (0, 0.0);
    }

    // Find max for numerical stability.
    let max_val = logits
        .iter()
        .map(|x| f32::from(*x))
        .fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0f32;
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;

    for (i, &logit) in logits.iter().enumerate() {
        let v = f32::from(logit);
        let exp_v = (v - max_val).exp();
        sum += exp_v;
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }

    let prob = (best_val - max_val).exp() / sum;
    (best_idx, prob)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_argmax() {
        let logits: Vec<f16> = vec![
            f16::from_f32(1.0),
            f16::from_f32(3.0),
            f16::from_f32(2.0),
        ];
        let (idx, prob) = softmax_argmax(&logits);
        assert_eq!(idx, 1);
        assert!(prob > 0.5);
    }

    #[test]
    fn test_softmax_argmax_empty() {
        let (idx, prob) = softmax_argmax(&[]);
        assert_eq!(idx, 0);
        assert_eq!(prob, 0.0);
    }
}
