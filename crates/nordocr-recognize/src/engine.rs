use std::path::Path;

use half::f16;
use nordocr_core::{OcrError, Result};
use nordocr_gpu::{GpuBuffer, GpuContext};
use nordocr_trt::{TrtEngine, TrtExecutionContext, TrtRuntime};

use crate::charset::CHARSET_SIZE;

/// PARSeq text recognition engine backed by TensorRT.
///
/// PARSeq is a non-autoregressive model that decodes all character
/// positions in parallel, giving much better GPU utilization than
/// autoregressive models like TrOCR.
///
/// Input: batch of text line images [B, 3, H, W] in FP16.
/// Output: logits [B, max_seq_len, charset_size] in FP16.
pub struct RecognitionEngine {
    context: TrtExecutionContext,
    input_buffer: GpuBuffer<f16>,
    output_buffer: GpuBuffer<f16>,
    /// Height text line images are resized to.
    input_height: u32,
    /// Maximum width of text line images.
    max_input_width: u32,
    /// Maximum sequence length (number of output characters).
    max_seq_len: u32,
    /// Maximum batch size.
    max_batch_size: u32,
}

impl RecognitionEngine {
    /// Load a pre-built TensorRT engine for PARSeq recognition.
    pub fn load(
        ctx: &GpuContext,
        engine_path: &Path,
        max_batch_size: u32,
        input_height: u32,
        max_input_width: u32,
        max_seq_len: u32,
    ) -> Result<Self> {
        let runtime = TrtRuntime::new()?;
        let engine = TrtEngine::load(&runtime, engine_path)?;
        let mut context = engine.create_context()?;

        // Input: [batch, 3, H, W] — 3-channel images.
        let input_size =
            (max_batch_size * 3 * input_height * max_input_width) as usize;
        // Output: [batch, max_seq_len, charset_size] — logits.
        let output_size =
            (max_batch_size * max_seq_len * CHARSET_SIZE as u32) as usize;

        let input_buffer = ctx.memory_pool.alloc::<f16>(input_size)?;
        let output_buffer = ctx.memory_pool.alloc::<f16>(output_size)?;

        context.set_tensor_address("input", &input_buffer)?;
        context.set_tensor_address("logits", &output_buffer)?;

        tracing::info!(
            engine = %engine_path.display(),
            input_h = input_height,
            max_w = max_input_width,
            max_seq = max_seq_len,
            max_batch = max_batch_size,
            "loaded recognition engine"
        );

        Ok(Self {
            context,
            input_buffer,
            output_buffer,
            input_height,
            max_input_width,
            max_seq_len,
            max_batch_size,
        })
    }

    /// Run recognition inference on a batch of text line images.
    ///
    /// Returns logits buffer (still on GPU).
    pub fn infer(
        &mut self,
        _line_images: &GpuBuffer<f16>,
        batch_size: u32,
        actual_width: u32,
        stream: u64,
    ) -> Result<&GpuBuffer<f16>> {
        if batch_size > self.max_batch_size {
            return Err(OcrError::Recognition(format!(
                "batch size {batch_size} exceeds max {}",
                self.max_batch_size
            )));
        }

        // Set dynamic shape for this batch.
        self.context.set_input_shape(
            "input",
            &[
                batch_size as i64,
                3,
                self.input_height as i64,
                actual_width.min(self.max_input_width) as i64,
            ],
        )?;

        self.context.enqueue_v3(stream)?;

        Ok(&self.output_buffer)
    }

    pub fn input_height(&self) -> u32 {
        self.input_height
    }

    pub fn max_input_width(&self) -> u32 {
        self.max_input_width
    }

    pub fn max_seq_len(&self) -> u32 {
        self.max_seq_len
    }

    pub fn max_batch_size(&self) -> u32 {
        self.max_batch_size
    }
}
