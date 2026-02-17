use half::f16;
use nordocr_core::{OcrError, Result};
use nordocr_gpu::{GpuBuffer, GpuContext};

use crate::charset::{self, CHARSET_SIZE, CTC_BLANK, CTC_NUM_CLASSES, EOS_TOKEN, PAD_TOKEN};

/// Decoded text with per-character confidence scores and timestep positions.
#[derive(Debug, Clone)]
pub struct DecodedText {
    pub text: String,
    pub confidence: f32,
    pub char_confidences: Vec<f32>,
    /// CTC timestep index where each character was first decoded.
    /// Timestep T corresponds to spatial position T * stride (stride=4 for SVTRv2).
    pub char_positions: Vec<u32>,
}

impl DecodedText {
    /// Trim trailing characters separated by a large spatial gap in the CTC output.
    ///
    /// Addresses trailing garbage from bbox extending into adjacent table cells.
    /// Uses timestep positions to detect spatial gaps: when the last few chars
    /// are decoded from timesteps far from the main body (large jump in position),
    /// they likely come from adjacent content that bled into the crop.
    ///
    /// Algorithm:
    /// 1. Compute gaps between consecutive char timestep positions.
    /// 2. Find the largest gap in the rightmost 40% of the text.
    /// 3. If it's >= 2× the median gap AND >= 3 timesteps (~12px), trim after it.
    /// 4. The trailing fragment must be short (≤ 6 chars, < 25% of total).
    pub fn trim_trailing_by_position(&mut self) {
        let n = self.char_positions.len();
        if n < 4 {
            return;
        }

        // Compute inter-character gaps in timestep space.
        let mut gaps: Vec<(usize, u32)> = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            let gap = self.char_positions[i + 1].saturating_sub(self.char_positions[i]);
            gaps.push((i, gap));
        }

        if gaps.is_empty() {
            return;
        }

        // Only look at the rightmost 40% of char positions for trailing gaps.
        let right_start = n * 60 / 100;

        // Find the largest gap in the right portion.
        let mut best_gap_idx = 0usize;
        let mut best_gap_val = 0u32;
        for &(i, g) in &gaps {
            if i >= right_start && g > best_gap_val {
                best_gap_val = g;
                best_gap_idx = i;
            }
        }

        // Must be at least 3 timesteps (~12px) to be a real spatial break.
        if best_gap_val < 3 {
            return;
        }

        // Compute median gap across ALL positions.
        let mut sorted_gaps: Vec<u32> = gaps.iter().map(|&(_, g)| g).collect();
        sorted_gaps.sort_unstable();
        let median_gap = sorted_gaps[sorted_gaps.len() / 2];

        // The trailing gap must be at least 2× the median gap.
        if median_gap > 0 && best_gap_val < median_gap * 2 {
            return;
        }

        // The trailing fragment (chars after the gap) must be small.
        let trailing_count = n - 1 - best_gap_idx;
        if trailing_count > 6 || trailing_count * 4 > n {
            return;
        }

        // Trim: keep chars up to (and including) best_gap_idx.
        let keep = best_gap_idx + 1;
        let chars: Vec<char> = self.text.chars().collect();

        // Also trim trailing spaces that are now exposed.
        let mut new_len = keep;
        while new_len > 0 && chars[new_len - 1] == ' ' {
            new_len -= 1;
        }

        if new_len < n && new_len >= 1 {
            self.text = chars[..new_len].iter().collect();
            self.char_confidences.truncate(new_len);
            self.char_positions.truncate(new_len);

            // Recompute overall confidence.
            if !self.char_confidences.is_empty() {
                let total_log: f32 = self.char_confidences.iter().map(|p| p.ln()).sum();
                self.confidence =
                    (total_log / self.char_confidences.len() as f32).exp();
            }
        }
    }
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
        let mut char_positions = Vec::new();
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
                char_positions.push(pos as u32);
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
            char_positions,
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

    /// Decode CTC output with per-item sequence lengths.
    ///
    /// Each item in the batch has its own width (`item_widths[i]`), so the CTC
    /// decode should only process `item_widths[i] / 4` timesteps for item i.
    /// The output tensor is laid out with `batch_seq_len` timesteps per item
    /// (padded to the batch-wide maximum).
    pub fn decode_cpu_per_item(
        &self,
        probs_cpu: &[f16],
        batch_size: u32,
        batch_seq_len: u32,
        item_widths: &[u32],
    ) -> Result<Vec<DecodedText>> {
        let stride = batch_seq_len as usize * self.num_classes;
        let mut results = Vec::with_capacity(batch_size as usize);

        for b in 0..batch_size as usize {
            let offset = b * stride;
            let end = offset + stride;
            let batch_probs = if end <= probs_cpu.len() {
                &probs_cpu[offset..end]
            } else {
                &[]
            };
            // Use this item's actual width to compute its seq_len.
            let item_seq_len = if b < item_widths.len() {
                item_widths[b] / 4
            } else {
                batch_seq_len
            };
            let decoded = self.decode_single_ctc(b as u32, batch_probs, item_seq_len)?;
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
        let mut char_positions = Vec::new();
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
                char_positions.push(t as u32);
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
            char_positions,
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
