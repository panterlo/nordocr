use std::path::Path;

use half::f16;
use nordocr_core::{OcrError, Result};
use nordocr_gpu::{GpuBuffer, GpuContext};
use nordocr_trt::{TrtEngine, TrtExecutionContext, TrtRuntime};

/// DBNet++ text detection engine backed by TensorRT.
///
/// Takes a preprocessed image and produces a probability map + threshold map,
/// which are combined via differentiable binarization to produce text region masks.
pub struct DetectionEngine {
    context: TrtExecutionContext,
    /// Input: [batch, 3, H, W] in FP16.
    input_buffer: GpuBuffer<f16>,
    /// Output probability map: [batch, 1, H, W] in FP16.
    prob_buffer: GpuBuffer<f16>,
    /// Output threshold map: [batch, 1, H, W] in FP16.
    thresh_buffer: GpuBuffer<f16>,
    /// Model input dimensions.
    input_height: u32,
    input_width: u32,
    max_batch_size: u32,
}

impl DetectionEngine {
    /// Load a pre-built TensorRT engine for DBNet++ detection.
    pub fn load(
        ctx: &GpuContext,
        engine_path: &Path,
        max_batch_size: u32,
        input_height: u32,
        input_width: u32,
    ) -> Result<Self> {
        let runtime = TrtRuntime::new()?;
        let engine = TrtEngine::load(&runtime, engine_path)?;
        let mut context = engine.create_context()?;

        // Allocate I/O buffers from the GPU memory pool.
        let input_size = (max_batch_size * 3 * input_height * input_width) as usize;
        let output_size = (max_batch_size * 1 * input_height * input_width) as usize;

        let input_buffer = ctx.memory_pool.alloc::<f16>(input_size)?;
        let prob_buffer = ctx.memory_pool.alloc::<f16>(output_size)?;
        let thresh_buffer = ctx.memory_pool.alloc::<f16>(output_size)?;

        // Bind buffers to TensorRT I/O tensors.
        context.set_tensor_address("input", &input_buffer)?;
        context.set_tensor_address("prob_map", &prob_buffer)?;
        context.set_tensor_address("thresh_map", &thresh_buffer)?;

        tracing::info!(
            engine = %engine_path.display(),
            input_h = input_height,
            input_w = input_width,
            max_batch = max_batch_size,
            "loaded detection engine"
        );

        Ok(Self {
            context,
            input_buffer,
            prob_buffer,
            thresh_buffer,
            input_height,
            input_width,
            max_batch_size,
        })
    }

    /// Run text detection on a batch of preprocessed images.
    ///
    /// Input images should be resized to (input_height, input_width) and
    /// normalized to the model's expected range (e.g., ImageNet normalization).
    ///
    /// Returns the probability map GPU buffer (still on GPU — no CPU copy).
    pub fn infer(
        &mut self,
        _preprocessed: &GpuBuffer<f16>,
        batch_size: u32,
        stream: u64,
    ) -> Result<&GpuBuffer<f16>> {
        if batch_size > self.max_batch_size {
            return Err(OcrError::Detection(format!(
                "batch size {batch_size} exceeds max {max}",
                max = self.max_batch_size
            )));
        }

        // Set dynamic batch dimension.
        self.context.set_input_shape(
            "input",
            &[
                batch_size as i64,
                3,
                self.input_height as i64,
                self.input_width as i64,
            ],
        )?;

        // Enqueue inference on the CUDA stream.
        self.context.enqueue_v3(stream)?;

        // Output stays on GPU — return reference to the probability map buffer.
        Ok(&self.prob_buffer)
    }

    pub fn input_height(&self) -> u32 {
        self.input_height
    }

    pub fn input_width(&self) -> u32 {
        self.input_width
    }

    pub fn max_batch_size(&self) -> u32 {
        self.max_batch_size
    }
}
