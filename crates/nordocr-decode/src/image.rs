use std::io::BufReader;

use nordocr_core::{OcrError, RawImage, Result};
use nordocr_gpu::{GpuBuffer, GpuContext};

/// Decodes image bytes (JPEG, PNG, TIFF) into GPU memory or CPU RawImage.
///
/// Prefers nvJPEG for hardware-accelerated JPEG decode directly to GPU.
/// Falls back to CPU decode via the `image` crate for other formats,
/// then uploads to GPU.
pub struct ImageDecoder {
    /// Whether nvJPEG is available for GPU-accelerated JPEG decode.
    nvjpeg_available: bool,
}

impl ImageDecoder {
    pub fn new(_ctx: &GpuContext) -> Result<Self> {
        Ok(Self {
            nvjpeg_available: false,
        })
    }

    /// Decode image bytes into a GPU buffer (grayscale).
    pub fn decode_to_gpu(
        &self,
        ctx: &GpuContext,
        data: &[u8],
    ) -> Result<(GpuBuffer<u8>, u32, u32)> {
        if self.nvjpeg_available && is_jpeg(data) {
            return self.decode_jpeg_gpu(ctx, data);
        }
        self.decode_cpu_upload(ctx, data)
    }

    /// Decode image bytes to CPU RGB `RawImage`.
    pub fn decode_to_rgb_cpu(&self, data: &[u8]) -> Result<RawImage> {
        let img = image::load_from_memory(data)
            .map_err(|e| OcrError::ImageDecode(e.to_string()))?;
        let rgb = img.to_rgb8();
        let width = rgb.width();
        let height = rgb.height();
        Ok(RawImage {
            data: rgb.into_raw(),
            width,
            height,
            channels: 3,
        })
    }

    /// Decode a multi-page TIFF to CPU RGB images (one per page).
    ///
    /// Uses the `tiff` crate to iterate frames. Supports common TIFF
    /// compressions (LZW, Deflate, PackBits). For Fax3/Fax4 (G4) TIFFs,
    /// use `decode_tiff_pages_via_image()` which may have broader support.
    pub fn decode_tiff_pages_cpu(
        &self,
        data: &[u8],
        page_filter: Option<&[u32]>,
    ) -> Result<Vec<RawImage>> {
        let cursor = std::io::Cursor::new(data);
        let mut decoder = tiff::decoder::Decoder::new(BufReader::new(cursor))
            .map_err(|e| OcrError::ImageDecode(format!("TIFF decode error: {e}")))?;

        let mut pages = Vec::new();
        let mut page_idx = 0u32;

        loop {
            let should_include = page_filter
                .map(|f| f.contains(&page_idx))
                .unwrap_or(true);

            if should_include {
                let (width, height) = decoder
                    .dimensions()
                    .map_err(|e| OcrError::ImageDecode(format!("TIFF dimensions: {e}")))?;
                let color_type = decoder
                    .colortype()
                    .map_err(|e| OcrError::ImageDecode(format!("TIFF colortype: {e}")))?;
                let image_data = decoder
                    .read_image()
                    .map_err(|e| OcrError::ImageDecode(format!("TIFF page {page_idx}: {e}")))?;

                let rgb = tiff_frame_to_rgb(&image_data, color_type, width, height)?;

                pages.push(RawImage {
                    data: rgb,
                    width,
                    height,
                    channels: 3,
                });
            }

            page_idx += 1;
            if decoder.next_image().is_err() {
                break;
            }
        }

        tracing::debug!(total_pages = page_idx, decoded = pages.len(), "TIFF decode");
        Ok(pages)
    }

    /// Decode a batch of images to GPU.
    pub fn decode_batch_to_gpu(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
    ) -> Result<Vec<(GpuBuffer<u8>, u32, u32)>> {
        images
            .iter()
            .map(|data| self.decode_to_gpu(ctx, data))
            .collect()
    }

    fn decode_jpeg_gpu(
        &self,
        ctx: &GpuContext,
        data: &[u8],
    ) -> Result<(GpuBuffer<u8>, u32, u32)> {
        self.decode_cpu_upload(ctx, data)
    }

    fn decode_cpu_upload(
        &self,
        ctx: &GpuContext,
        data: &[u8],
    ) -> Result<(GpuBuffer<u8>, u32, u32)> {
        let img = image::load_from_memory(data)
            .map_err(|e| OcrError::ImageDecode(e.to_string()))?;

        let gray = img.to_luma8();
        let width = gray.width();
        let height = gray.height();
        let pixels = gray.into_raw();

        let gpu_buf = ctx
            .default_stream
            .clone_htod(&pixels)
            .map_err(|e| OcrError::Cuda(format!("htod copy failed: {e}")))?;

        let buffer = GpuBuffer::from_cuda_slice(gpu_buf, pixels.len(), &ctx.default_stream);
        Ok((buffer, width, height))
    }
}

/// Convert a TIFF frame to RGB u8 data.
fn tiff_frame_to_rgb(
    data: &tiff::decoder::DecodingResult,
    color_type: tiff::ColorType,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    let n = (width * height) as usize;
    match data {
        tiff::decoder::DecodingResult::U8(buf) => match color_type {
            tiff::ColorType::Gray(8) | tiff::ColorType::Gray(1) => {
                let mut rgb = Vec::with_capacity(n * 3);
                for &g in buf.iter().take(n) {
                    rgb.push(g);
                    rgb.push(g);
                    rgb.push(g);
                }
                Ok(rgb)
            }
            tiff::ColorType::RGB(8) => Ok(buf.clone()),
            tiff::ColorType::RGBA(8) => {
                let mut rgb = Vec::with_capacity(n * 3);
                for chunk in buf.chunks_exact(4) {
                    rgb.push(chunk[0]);
                    rgb.push(chunk[1]);
                    rgb.push(chunk[2]);
                }
                Ok(rgb)
            }
            other => Err(OcrError::ImageDecode(format!(
                "unsupported TIFF color type: {:?}",
                other
            ))),
        },
        tiff::decoder::DecodingResult::U16(buf) => match color_type {
            tiff::ColorType::Gray(16) => {
                let mut rgb = Vec::with_capacity(n * 3);
                for &g in buf.iter().take(n) {
                    let g8 = (g >> 8) as u8;
                    rgb.push(g8);
                    rgb.push(g8);
                    rgb.push(g8);
                }
                Ok(rgb)
            }
            other => Err(OcrError::ImageDecode(format!(
                "unsupported 16-bit TIFF color type: {:?}",
                other
            ))),
        },
        _ => Err(OcrError::ImageDecode(
            "unsupported TIFF decoding result".into(),
        )),
    }
}

/// Convert a `RawImage` (already decoded on CPU) to GPU.
pub fn raw_image_to_gpu(ctx: &GpuContext, img: &RawImage) -> Result<GpuBuffer<u8>> {
    let gpu_buf = ctx
        .default_stream
        .clone_htod(&img.data)
        .map_err(|e| OcrError::Cuda(format!("htod copy failed: {e}")))?;
    Ok(GpuBuffer::from_cuda_slice(gpu_buf, img.data.len(), &ctx.default_stream))
}

fn is_jpeg(data: &[u8]) -> bool {
    data.len() >= 2 && data[0] == 0xFF && data[1] == 0xD8
}
