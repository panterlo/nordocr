use nordocr_core::{OcrError, RawImage, Result};
use nordocr_gpu::{GpuBuffer, GpuContext};

/// Decodes image bytes (JPEG, PNG, TIFF) into GPU memory.
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
        // In production: probe for nvJPEG library availability.
        // nvjpegCreate(&handle, ...)
        Ok(Self {
            nvjpeg_available: false, // set true when nvJPEG is linked
        })
    }

    /// Decode image bytes into a GPU buffer.
    ///
    /// Returns the decoded image as a grayscale u8 buffer on GPU,
    /// along with dimensions.
    pub fn decode_to_gpu(
        &self,
        ctx: &GpuContext,
        data: &[u8],
    ) -> Result<(GpuBuffer<u8>, u32, u32)> {
        // Detect format from magic bytes.
        if self.nvjpeg_available && is_jpeg(data) {
            return self.decode_jpeg_gpu(ctx, data);
        }

        // CPU fallback for all formats.
        self.decode_cpu_upload(ctx, data)
    }

    /// Decode a batch of images to GPU (for multi-page documents).
    pub fn decode_batch_to_gpu(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
    ) -> Result<Vec<(GpuBuffer<u8>, u32, u32)>> {
        // In production with nvJPEG:
        //   Use nvjpegDecodeBatched for parallel JPEG decode.
        //   This overlaps CPU parsing with GPU IDCT across images.
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
        // In production:
        //   nvjpegDecode(handle, state, data.as_ptr(), data.len(),
        //                NVJPEG_OUTPUT_Y,  // grayscale
        //                &image, stream)
        //   The decoded image lands directly in GPU memory — zero CPU copy.

        // Fallback: decode on CPU and upload.
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

        // Upload to GPU.
        let gpu_buf = ctx
            .device
            .htod_sync_copy(&pixels)
            .map_err(|e| OcrError::Cuda(format!("htod copy failed: {e}")))?;

        let buffer = GpuBuffer::from_cuda_slice(gpu_buf, pixels.len());

        Ok((buffer, width, height))
    }
}

/// Convert a `RawImage` (already decoded on CPU) to GPU.
pub fn raw_image_to_gpu(ctx: &GpuContext, img: &RawImage) -> Result<GpuBuffer<u8>> {
    let gpu_buf = ctx
        .device
        .htod_sync_copy(&img.data)
        .map_err(|e| OcrError::Cuda(format!("htod copy failed: {e}")))?;
    Ok(GpuBuffer::from_cuda_slice(gpu_buf, img.data.len()))
}

fn is_jpeg(data: &[u8]) -> bool {
    data.len() >= 2 && data[0] == 0xFF && data[1] == 0xD8
}
