pub mod image;
pub mod pdf;

pub use self::image::ImageDecoder;
pub use pdf::PdfDecoder;

use nordocr_core::{FileInput, OcrError, RawImage, Result};
use nordocr_gpu::{GpuBuffer, GpuContext};

/// Unified decoder that handles all supported input formats.
pub struct Decoder {
    image_decoder: ImageDecoder,
    pdf_decoder: PdfDecoder,
}

impl Decoder {
    pub fn new(ctx: &GpuContext) -> Result<Self> {
        Ok(Self {
            image_decoder: ImageDecoder::new(ctx)?,
            pdf_decoder: PdfDecoder::new(300.0), // 300 DPI default for OCR
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
