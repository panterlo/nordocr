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
pub struct Decoder {
    image_decoder: ImageDecoder,
    pdf_decoder: PdfDecoder,

    #[cfg(feature = "dali")]
    dali_decoder: Option<DaliDecoder>,
}

impl Decoder {
    pub fn new(ctx: &GpuContext) -> Result<Self> {
        Ok(Self {
            image_decoder: ImageDecoder::new(ctx)?,
            pdf_decoder: PdfDecoder::new(300.0),

            #[cfg(feature = "dali")]
            dali_decoder: match DaliDecoder::new(ctx) {
                Ok(d) => {
                    tracing::info!("DALI decoder initialized");
                    Some(d)
                }
                Err(e) => {
                    tracing::warn!("DALI decoder unavailable: {e}");
                    None
                }
            },
        })
    }

    /// Decode a file input into one or more GPU buffers (one per page).
    ///
    /// Returns: Vec of (gpu_buffer, width, height) per page.
    pub fn decode(
        &self,
        ctx: &GpuContext,
        input: &FileInput,
        page_filter: Option<&[u32]>,
    ) -> Result<Vec<(GpuBuffer<u8>, u32, u32)>> {
        match input {
            FileInput::Image(data) => {
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
                // Decode via tiff crate for multi-page support, then upload each page.
                let raw_images = self.image_decoder.decode_tiff_pages_cpu(data, page_filter)?;
                if raw_images.is_empty() {
                    // Fallback to image crate for formats tiff crate doesn't support (e.g. Fax4).
                    tracing::warn!("tiff crate returned no pages, falling back to image crate (first page only)");
                    let result = self.image_decoder.decode_to_gpu(ctx, data)?;
                    return Ok(vec![result]);
                }
                raw_images
                    .iter()
                    .map(|img| {
                        let buf = self::image::raw_image_to_gpu(ctx, img)?;
                        Ok((buf, img.width, img.height))
                    })
                    .collect()
            }
        }
    }

    /// Decode a file input into CPU RGB images (no GPU upload).
    ///
    /// Used by CPU-based detection (morphological) to avoid unnecessary GPU transfers.
    pub fn decode_cpu(
        &self,
        input: &FileInput,
        page_filter: Option<&[u32]>,
    ) -> Result<Vec<RawImage>> {
        match input {
            FileInput::Image(data) => {
                let img = self.image_decoder.decode_to_rgb_cpu(data)?;
                Ok(vec![img])
            }
            FileInput::Pdf(data) => self.pdf_decoder.render_pages(data, page_filter),
            FileInput::MultiPageTiff(data) => {
                let pages = self.image_decoder.decode_tiff_pages_cpu(data, page_filter)?;
                if pages.is_empty() {
                    // Fallback for unsupported TIFF compressions.
                    tracing::warn!("tiff crate returned no pages, falling back to image crate");
                    let img = self.image_decoder.decode_to_rgb_cpu(data)?;
                    return Ok(vec![img]);
                }
                Ok(pages)
            }
        }
    }
}
