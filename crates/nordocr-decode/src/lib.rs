pub mod image;
pub mod pdf;

#[cfg(feature = "dali")]
pub mod dali;

pub use self::image::ImageDecoder;
pub use pdf::PdfDecoder;

#[cfg(feature = "dali")]
pub use dali::DaliDecoder;

use nordocr_core::{FileInput, OcrError, RawImage, Result};
use nordocr_gpu::{GpuBuffer, GpuContext};

/// Unified decoder that handles all supported input formats.
///
/// When the `dali` feature is enabled and a [`DaliDecoder`] is available,
/// batch image decodes are routed through the NVIDIA DALI pipeline for
/// GPU-accelerated fused decode + resize + normalize. Otherwise, the standard
/// [`ImageDecoder`] (nvJPEG / CPU fallback) path is used.
pub struct Decoder {
    image_decoder: ImageDecoder,
    pdf_decoder: PdfDecoder,

    /// Optional DALI-backed decoder for GPU-accelerated batch image decode.
    /// Preferred over `image_decoder` for batch operations when available.
    #[cfg(feature = "dali")]
    dali_decoder: Option<DaliDecoder>,
}

impl Decoder {
    pub fn new(ctx: &GpuContext) -> Result<Self> {
        Ok(Self {
            image_decoder: ImageDecoder::new(ctx)?,
            pdf_decoder: PdfDecoder::new(300.0), // 300 DPI default for OCR

            #[cfg(feature = "dali")]
            dali_decoder: match DaliDecoder::new(ctx) {
                Ok(d) => {
                    tracing::info!("DALI decoder initialized; batch decodes will use GPU-accelerated pipeline");
                    Some(d)
                }
                Err(e) => {
                    tracing::warn!("DALI decoder unavailable, falling back to standard decode: {e}");
                    None
                }
            },
        })
    }

    /// Decode a file input into one or more GPU buffers (one per page).
    ///
    /// Returns: Vec of (gpu_buffer, width, height) per page.
    ///
    /// When the `dali` feature is enabled and a DALI decoder is available,
    /// image inputs are decoded through the DALI pipeline for higher throughput.
    pub fn decode(
        &self,
        ctx: &GpuContext,
        input: &FileInput,
        page_filter: Option<&[u32]>,
    ) -> Result<Vec<(GpuBuffer<u8>, u32, u32)>> {
        match input {
            FileInput::Image(data) => {
                // When DALI is available, prefer it even for single images
                // so the decode goes through the GPU-accelerated pipeline.
                #[cfg(feature = "dali")]
                if let Some(ref dali) = self.dali_decoder {
                    return dali.decode_batch_to_gpu(ctx, &[data.to_vec()]);
                }

                let result = self.image_decoder.decode_to_gpu(ctx, data)?;
                Ok(vec![result])
            }
            FileInput::Pdf(data) => {
                let raw_images = self.pdf_decoder.render_pages(data, page_filter)?;
                raw_images
                    .iter()
                    .map(|img| {
                        let buf = self::image::raw_image_to_gpu(ctx, img)?;
                        Ok((buf, img.width, img.height))
                    })
                    .collect()
            }
            FileInput::MultiPageTiff(data) => {
                // Decode each TIFF page as a separate image.
                // The `image` crate handles multi-frame TIFF.
                let result = self.image_decoder.decode_to_gpu(ctx, data)?;
                Ok(vec![result])
            }
        }
    }
}
