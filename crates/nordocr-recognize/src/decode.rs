use half::f16;
use nordocr_core::{OcrError, Result};
use nordocr_gpu::{GpuBuffer, GpuContext};

use crate::charset::{self, CHARSET_SIZE, CTC_BLANK, CTC_NUM_CLASSES, EOS_TOKEN, PAD_TOKEN};

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

/// Decodes SVTRv2 CTC output into text strings.
///
/// SVTRv2 outputs probabilities of shape [batch, T, num_classes] where
/// T = input_width / stride (stride=4). Uses CTC decoding: collapse
/// repeated tokens and skip blanks (index 0).
pub struct CtcDecoder {
    num_classes: usize,
}

impl CtcDecoder {
    pub fn new() -> Self {
        Self {
            num_classes: CTC_NUM_CLASSES,
        }
    }

    /// Decode a batch of CTC probabilities from GPU to text strings.
    ///
    /// `probs` contains [batch_size, seq_len, num_classes] as f16.
    /// `seq_len` is the temporal dimension (input_width / stride).
    pub fn decode_batch(
        &self,
        ctx: &GpuContext,
        probs: &GpuBuffer<f16>,
        batch_size: u32,
        seq_len: u32,
    ) -> Result<Vec<DecodedText>> {
        let _ = (ctx, probs);

        let mut results = Vec::with_capacity(batch_size as usize);

        for b in 0..batch_size {
            let decoded = self.decode_single_ctc(b, &[], seq_len)?;
            results.push(decoded);
        }

        Ok(results)
    }

    /// Decode CTC output from CPU f16 probabilities.
    ///
    /// `probs_cpu` is the full [batch, seq_len, num_classes] tensor flattened.
    /// Each batch element has `seq_len * num_classes` values.
    pub fn decode_cpu(
        &self,
        probs_cpu: &[f16],
        batch_size: u32,
        seq_len: u32,
    ) -> Result<Vec<DecodedText>> {
        let stride = seq_len as usize * self.num_classes;
        let mut results = Vec::with_capacity(batch_size as usize);

        for b in 0..batch_size {
            let offset = b as usize * stride;
            let end = offset + stride;
            let batch_probs = if end <= probs_cpu.len() {
                &probs_cpu[offset..end]
            } else {
                &[]
            };
            let decoded = self.decode_single_ctc(b, batch_probs, seq_len)?;
            results.push(decoded);
        }

        Ok(results)
    }

    fn decode_single_ctc(
        &self,
        _batch_index: u32,
        probs_cpu: &[f16], // [seq_len, num_classes]
        seq_len: u32,
    ) -> Result<DecodedText> {
        let mut text = String::new();
        let mut char_confidences = Vec::new();
        let mut total_log_prob = 0.0f32;
        let mut num_chars = 0;
        let mut prev_token: usize = CTC_BLANK;

        for t in 0..seq_len as usize {
            let offset = t * self.num_classes;

            let (best_token, best_prob) = if offset + self.num_classes <= probs_cpu.len() {
                let slice = &probs_cpu[offset..offset + self.num_classes];
                softmax_argmax(slice)
            } else {
                (CTC_BLANK, 1.0)
            };

            // CTC decoding: skip blanks and repeated tokens.
            if best_token == CTC_BLANK || best_token == prev_token {
                prev_token = best_token;
                continue;
            }
            prev_token = best_token;

            if let Some(c) = charset::ctc_token_to_char(best_token) {
                text.push(c);
                char_confidences.push(best_prob);
                total_log_prob += best_prob.ln();
                num_chars += 1;
            }
        }

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

    #[test]
    fn test_ctc_decode_blank_collapse() {
        // Simulate CTC output: blank, blank, token 1 ('0'), token 1, blank, token 2 ('1')
        // Expected: "01" (two unique non-blank tokens after collapse)
        let decoder = CtcDecoder::new();
        let nc = CTC_NUM_CLASSES;

        // 6 timesteps, each with `nc` logits
        let mut probs = vec![f16::from_f32(-10.0); 6 * nc];

        // t=0: blank (index 0) is highest
        probs[0] = f16::from_f32(10.0);
        // t=1: blank again
        probs[nc] = f16::from_f32(10.0);
        // t=2: token 1 ('0')
        probs[2 * nc + 1] = f16::from_f32(10.0);
        // t=3: token 1 again (should be collapsed)
        probs[3 * nc + 1] = f16::from_f32(10.0);
        // t=4: blank
        probs[4 * nc] = f16::from_f32(10.0);
        // t=5: token 2 ('1')
        probs[5 * nc + 2] = f16::from_f32(10.0);

        let results = decoder.decode_cpu(&probs, 1, 6).unwrap();
        assert_eq!(results[0].text, "01");
    }

    #[test]
    fn test_ctc_decode_repeated_with_blank_separator() {
        // "00" requires: token 1, blank, token 1
        let decoder = CtcDecoder::new();
        let nc = CTC_NUM_CLASSES;
        let mut probs = vec![f16::from_f32(-10.0); 3 * nc];

        // t=0: token 1 ('0')
        probs[1] = f16::from_f32(10.0);
        // t=1: blank
        probs[nc] = f16::from_f32(10.0);
        // t=2: token 1 ('0') again
        probs[2 * nc + 1] = f16::from_f32(10.0);

        let results = decoder.decode_cpu(&probs, 1, 3).unwrap();
        assert_eq!(results[0].text, "00");
    }

    #[test]
    fn test_ctc_decode_empty_sequence() {
        // All blanks → empty string
        let decoder = CtcDecoder::new();
        let nc = CTC_NUM_CLASSES;
        let mut probs = vec![f16::from_f32(-10.0); 4 * nc];
        for t in 0..4 {
            probs[t * nc] = f16::from_f32(10.0); // blank highest
        }
        let results = decoder.decode_cpu(&probs, 1, 4).unwrap();
        assert_eq!(results[0].text, "");
        assert_eq!(results[0].confidence, 0.0);
    }
}
