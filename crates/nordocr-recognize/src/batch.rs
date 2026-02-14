use nordocr_core::{BBox, Result, TextLine, TextRegion};
use nordocr_gpu::GpuContext;

use crate::decode::{DecodedText, TokenDecoder};
use crate::engine::RecognitionEngine;

/// Batches text line images for efficient recognition inference.
///
/// Key optimization: sort lines by width before batching to minimize
/// padding waste. Lines with similar widths are grouped together,
/// so the batch tensor has minimal unused space.
pub struct RecognitionBatcher {
    engine: RecognitionEngine,
    decoder: TokenDecoder,
    /// Maximum number of lines per batch.
    batch_size: u32,
}

/// A prepared batch of text line images ready for inference.
struct LineBatch {
    /// Indices into the original regions array (for result reassembly).
    region_indices: Vec<usize>,
    /// Actual width of images in this batch (all padded to this width).
    batch_width: u32,
    /// Number of lines in this batch.
    count: u32,
}

impl RecognitionBatcher {
    pub fn new(engine: RecognitionEngine, batch_size: u32) -> Self {
        let max_seq_len = engine.max_seq_len();
        Self {
            engine,
            decoder: TokenDecoder::new(max_seq_len),
            batch_size,
        }
    }

    /// Recognize text in all detected regions across all pages.
    ///
    /// 1. Crop text line images from the preprocessed page images.
    /// 2. Sort by width to minimize padding.
    /// 3. Batch into groups of `batch_size`.
    /// 4. Run PARSeq inference on each batch.
    /// 5. Decode tokens → text.
    /// 6. Reassemble results in original order.
    pub fn recognize_all(
        &mut self,
        ctx: &GpuContext,
        regions: &[TextRegion],
        stream: u64,
    ) -> Result<Vec<TextLine>> {
        if regions.is_empty() {
            return Ok(Vec::new());
        }

        // Create width-sorted batches.
        let batches = self.create_batches(regions);

        let mut results: Vec<Option<TextLine>> = vec![None; regions.len()];

        for batch in &batches {
            // In production:
            // 1. Crop and resize line images from page buffers to batch tensor on GPU.
            //    - All lines in this batch are padded to `batch.batch_width`.
            //    - This uses a resize+pad CUDA kernel from the memory pool.
            //
            // 2. Run recognition inference.
            //    let logits = self.engine.infer(&batch_tensor, batch.count, batch.batch_width, stream)?;
            //
            // 3. Decode logits to text.
            //    let decoded = self.decoder.decode_batch(ctx, logits, batch.count)?;

            // Placeholder: create empty results.
            let decoded: Vec<DecodedText> = (0..batch.count)
                .map(|_| DecodedText {
                    text: String::new(),
                    confidence: 0.0,
                    char_confidences: Vec::new(),
                })
                .collect();

            // Map results back to original order.
            for (i, decoded_text) in decoded.into_iter().enumerate() {
                let orig_idx = batch.region_indices[i];
                let region = &regions[orig_idx];

                results[orig_idx] = Some(TextLine {
                    text: decoded_text.text,
                    confidence: decoded_text.confidence,
                    bbox: region.bbox,
                    words: None, // word-level boxes computed on request
                });
            }
        }

        let _ = (ctx, stream);

        Ok(results.into_iter().flatten().collect())
    }

    /// Sort regions by width and group into fixed-size batches.
    fn create_batches(&self, regions: &[TextRegion]) -> Vec<LineBatch> {
        // Create index-width pairs and sort by width.
        let mut indexed: Vec<(usize, u32)> = regions
            .iter()
            .enumerate()
            .map(|(i, r)| (i, r.bbox.width as u32))
            .collect();

        indexed.sort_by_key(|&(_, w)| w);

        // Group into batches.
        let mut batches = Vec::new();
        for chunk in indexed.chunks(self.batch_size as usize) {
            let max_width = chunk.iter().map(|&(_, w)| w).max().unwrap_or(0);
            let max_width = max_width.min(self.engine.max_input_width());

            batches.push(LineBatch {
                region_indices: chunk.iter().map(|&(i, _)| i).collect(),
                batch_width: max_width,
                count: chunk.len() as u32,
            });
        }

        tracing::debug!(
            total_lines = regions.len(),
            num_batches = batches.len(),
            batch_size = self.batch_size,
            "created recognition batches"
        );

        batches
    }
}
