use half::f16;
use nordocr_core::{RawImage, Result, TextLine, TextRegion};
use nordocr_gpu::GpuContext;

use crate::decode::{CtcDecoder, DecodedText};
use crate::engine::RecognitionEngine;

/// Batches text line images for efficient recognition inference.
///
/// Key optimization: sort lines by width before batching to minimize
/// padding waste. Lines with similar widths are grouped together,
/// so the batch tensor has minimal unused space.
pub struct RecognitionBatcher {
    engine: RecognitionEngine,
    decoder: CtcDecoder,
    /// Maximum number of lines per batch.
    batch_size: u32,
}

/// A prepared batch of text line images ready for inference.
struct LineBatch {
    /// Indices into the original regions array (for result reassembly).
    region_indices: Vec<usize>,
    /// Per-item actual widths (before padding to batch_width).
    item_widths: Vec<u32>,
    /// Actual width of images in this batch (all padded to this width).
    batch_width: u32,
    /// Number of lines in this batch.
    count: u32,
}

impl RecognitionBatcher {
    pub fn new(engine: RecognitionEngine, batch_size: u32) -> Self {
        Self {
            engine,
            decoder: CtcDecoder::new(),
            batch_size,
        }
    }

    /// Recognize text in all detected regions across all pages.
    ///
    /// `page_images` provides the CPU RGB image data for each page (indexed by page_index).
    ///
    /// 1. Crop text line images from the page images using region bboxes.
    /// 2. Sort by width to minimize padding.
    /// 3. Batch into groups of `batch_size`.
    /// 4. For each batch: resize + normalize + upload + TRT inference.
    /// 5. CTC decode tokens → text.
    /// 6. Reassemble results in original order.
    pub fn recognize_all(
        &mut self,
        _ctx: &GpuContext,
        regions: &[TextRegion],
        page_images: &[RawImage],
    ) -> Result<Vec<TextLine>> {
        if regions.is_empty() {
            return Ok(Vec::new());
        }

        // Pre-process all regions: crop, resize, normalize.
        let input_h = self.engine.input_height();
        let max_w = self.engine.max_input_width();

        let prepared: Vec<PreparedLine> = regions
            .iter()
            .enumerate()
            .map(|(i, region)| {
                let page = page_images
                    .get(region.page_index as usize)
                    .expect("page_index out of range");
                prepare_line(page, &region.bbox, input_h, max_w, i)
            })
            .collect();

        // Create width-sorted batches.
        let batches = self.create_batches(&prepared);

        let mut results: Vec<Option<TextLine>> = vec![None; regions.len()];

        for batch in &batches {
            // Build the batch tensor [B, 3, H, W] as contiguous f16.
            let b = batch.count as usize;
            let h = input_h as usize;
            let w = batch.batch_width as usize;
            let batch_elements = b * 3 * h * w;
            let mut batch_f16 = vec![f16::ZERO; batch_elements];

            for (batch_idx, &orig_idx) in batch.region_indices.iter().enumerate() {
                let line = &prepared[orig_idx];
                let line_w = line.width as usize;
                // Copy line data into the batch tensor (pad right with zeros).
                for c in 0..3 {
                    for y in 0..h {
                        let src_offset = c * h * line_w + y * line_w;
                        let dst_offset = batch_idx * 3 * h * w + c * h * w + y * w;
                        let copy_w = line_w.min(w);
                        batch_f16[dst_offset..dst_offset + copy_w]
                            .copy_from_slice(&line.data[src_offset..src_offset + copy_w]);
                    }
                }
            }

            // Run TRT inference.
            let output_f16 = self.engine.infer_batch_cpu(
                &batch_f16,
                batch.count,
                batch.batch_width,
            )?;

            // CTC decode: per-item seq_len to avoid decoding padded positions.
            // The batch tensor is padded to batch_width, but each item's actual
            // content occupies only item_widths[i] pixels. Decoding beyond that
            // produces garbage from the zero-padded region.
            let batch_seq_len = batch.batch_width / 4; // SVTRv2 stride = 4
            let decoded = self.decoder.decode_cpu_per_item(
                &output_f16,
                batch.count,
                batch_seq_len,
                &batch.item_widths,
            )?;

            // Map results back to original order.
            for (i, mut decoded_text) in decoded.into_iter().enumerate() {
                let orig_idx = batch.region_indices[i];
                let region = &regions[orig_idx];

                // Trim trailing garbage by spatial gap analysis (catches
                // cases where adjacent content bleeds into the crop itself).
                decoded_text.trim_trailing_by_position();

                // Strip leading/trailing whitespace from decoded text.
                // Handles cases where the model reads bbox whitespace as spaces.
                decoded_text.strip_whitespace();

                results[orig_idx] = Some(TextLine {
                    text: decoded_text.text,
                    confidence: decoded_text.confidence,
                    bbox: region.bbox,
                    words: None,
                    char_confidences: decoded_text.char_confidences,
                });
            }
        }

        Ok(results.into_iter().flatten().collect())
    }

    /// Sort regions by width and group into fixed-size batches.
    fn create_batches(&self, prepared: &[PreparedLine]) -> Vec<LineBatch> {
        let mut indexed: Vec<(usize, u32)> = prepared
            .iter()
            .map(|p| (p.orig_index, p.width))
            .collect();

        indexed.sort_by_key(|&(_, w)| w);

        let mut batches = Vec::new();
        for chunk in indexed.chunks(self.batch_size as usize) {
            let max_width = chunk.iter().map(|&(_, w)| w).max().unwrap_or(0);
            let max_width = max_width.min(self.engine.max_input_width());

            batches.push(LineBatch {
                region_indices: chunk.iter().map(|&(i, _)| i).collect(),
                item_widths: chunk.iter().map(|&(_, w)| w.min(max_width)).collect(),
                batch_width: max_width,
                count: chunk.len() as u32,
            });
        }

        tracing::debug!(
            total_lines = prepared.len(),
            num_batches = batches.len(),
            batch_size = self.batch_size,
            "created recognition batches"
        );

        batches
    }
}

/// A pre-processed text line ready for batching.
struct PreparedLine {
    /// CHW f16 data [3, H, W] normalized to [-1, 1].
    data: Vec<f16>,
    /// Width after resize (aligned to 4).
    width: u32,
    /// Original index in the regions array.
    orig_index: usize,
}

/// Crop a text region from a page image, resize to recognition input size,
/// normalize to [-1, 1], and convert to f16 CHW.
fn prepare_line(
    page: &RawImage,
    bbox: &nordocr_core::BBox,
    target_h: u32,
    max_w: u32,
    orig_index: usize,
) -> PreparedLine {
    let pw = page.width as usize;
    let ph = page.height as usize;
    let channels = page.channels as usize;

    // Clamp bbox to page bounds.
    let x0 = (bbox.x.max(0.0) as usize).min(pw.saturating_sub(1));
    let y0 = (bbox.y.max(0.0) as usize).min(ph.saturating_sub(1));
    let x1 = ((bbox.x + bbox.width).ceil() as usize).min(pw);
    let y1 = ((bbox.y + bbox.height).ceil() as usize).min(ph);

    let crop_w = x1.saturating_sub(x0).max(1);
    let crop_h = y1.saturating_sub(y0).max(1);

    // Calculate resize dimensions (aspect-preserving to target_h).
    let scale = target_h as f64 / crop_h as f64;
    let new_w = ((crop_w as f64 * scale).round() as u32).max(4);
    let new_w = (new_w / 4) * 4; // align to stride 4
    let new_w = new_w.min(max_w).max(4);
    let new_h = target_h;

    // Bilinear resize + normalize + HWC→CHW + u8→f16.
    let mut chw_f16 = vec![f16::ZERO; 3 * new_h as usize * new_w as usize];

    for dy in 0..new_h as usize {
        for dx in 0..new_w as usize {
            // Map destination pixel to source pixel (bilinear).
            let sx = dx as f64 * crop_w as f64 / new_w as f64;
            let sy = dy as f64 * crop_h as f64 / new_h as f64;

            let sx0 = (sx as usize).min(crop_w.saturating_sub(1));
            let sy0 = (sy as usize).min(crop_h.saturating_sub(1));
            let sx1 = (sx0 + 1).min(crop_w.saturating_sub(1));
            let sy1 = (sy0 + 1).min(crop_h.saturating_sub(1));

            let fx = sx - sx0 as f64;
            let fy = sy - sy0 as f64;

            for c in 0..3 {
                let src_c = if channels >= 3 { c } else { 0 }; // handle grayscale
                let get_pixel = |px: usize, py: usize| -> f64 {
                    let idx = (y0 + py) * pw * channels + (x0 + px) * channels + src_c;
                    page.data.get(idx).copied().unwrap_or(128) as f64
                };

                let v00 = get_pixel(sx0, sy0);
                let v10 = get_pixel(sx1, sy0);
                let v01 = get_pixel(sx0, sy1);
                let v11 = get_pixel(sx1, sy1);

                let v = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;

                // Normalize: (v / 255.0 - 0.5) / 0.5 = v / 127.5 - 1.0
                let normalized = (v / 127.5 - 1.0) as f32;

                // CHW layout: [C, H, W]
                let dst_idx = c * (new_h as usize * new_w as usize)
                    + dy * new_w as usize
                    + dx;
                chw_f16[dst_idx] = f16::from_f32(normalized);
            }
        }
    }

    PreparedLine {
        data: chw_f16,
        width: new_w,
        orig_index,
    }
}
