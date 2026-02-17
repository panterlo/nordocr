use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaStream, DevicePtr};
use half::f16;
use nordocr_core::{OcrError, Result};
use nordocr_gpu::GpuContext;
use nordocr_trt::{TrtEngine, TrtExecutionContext, TrtRuntime};

/// Text recognition engine backed by TensorRT.
///
/// Supports both SVTRv2 (CTC, variable width) and PARSeq (AR, fixed width).
///
/// Input: batch of text line images [B, 3, H, W] in FP16.
/// Output: logits/probs [B, T, C] in FP16 (or FP32 depending on engine).
pub struct RecognitionEngine {
    context: TrtExecutionContext,
    /// Pre-allocated output buffer (as u8 bytes for easy dtoh).
    output_slice: cudarc::driver::CudaSlice<u8>,
    output_ptr: u64,
    /// Output tensor name from the engine.
    output_name: String,
    /// Input tensor name from the engine.
    input_name: String,
    /// Height text line images are resized to.
    input_height: u32,
    /// Maximum width of text line images.
    max_input_width: u32,
    /// Maximum sequence length (number of output positions).
    max_seq_len: u32,
    /// Maximum batch size.
    max_batch_size: u32,
    /// Number of output classes per position.
    num_classes: u32,
    /// Size of one output element in bytes (2 for f16, 4 for f32).
    output_element_size: usize,
    /// Reference to the CUDA stream for upload/download.
    stream: Arc<CudaStream>,
}

impl RecognitionEngine {
    /// Load a pre-built TensorRT engine for text recognition.
    ///
    /// Reads I/O tensor names from the engine instead of hardcoding them.
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

        // Read actual tensor names from the engine.
        let input_name = engine
            .input_names()
            .first()
            .ok_or_else(|| OcrError::ModelLoad("engine has no input tensors".into()))?
            .clone();
        let output_name = engine
            .output_names()
            .first()
            .ok_or_else(|| OcrError::ModelLoad("engine has no output tensors".into()))?
            .clone();

        let output_dtype = engine.output_dtypes().first().copied();

        let mut context = engine.create_context()?;

        // Determine output element size based on dtype.
        let output_element_size = match output_dtype {
            Some(nordocr_trt::TrtDataType::Float16) => 2,
            Some(nordocr_trt::TrtDataType::Float32) => 4,
            _ => 2, // default to f16
        };

        // Figure out actual num_classes from the engine output shape.
        // SVTRv2: output is [B, T, 126], PARSeq: output is [B, max_seq_len, 126]
        // For now use the charset size constant; the engine shape is dynamic on B.
        let num_classes = crate::charset::CTC_NUM_CLASSES as u32;

        // Pre-allocate output buffer as raw u8 bytes.
        let output_size_bytes =
            (max_batch_size * max_seq_len * num_classes) as usize * output_element_size;
        let output_slice = ctx
            .default_stream
            .alloc_zeros::<u8>(output_size_bytes)
            .map_err(|e| OcrError::Cuda(format!("output alloc failed: {e}")))?;
        let output_ptr = {
            let (ptr, _sync) = DevicePtr::device_ptr(&output_slice, &ctx.default_stream);
            ptr as u64
        };

        // Bind output tensor.
        context.set_tensor_address_raw(&output_name, output_ptr)?;

        tracing::info!(
            engine = %engine_path.display(),
            input = %input_name,
            output = %output_name,
            input_h = input_height,
            max_w = max_input_width,
            max_seq = max_seq_len,
            max_batch = max_batch_size,
            num_classes,
            "loaded recognition engine"
        );

        Ok(Self {
            context,
            output_slice,
            output_ptr,
            output_name,
            input_name,
            input_height,
            max_input_width,
            max_seq_len,
            max_batch_size,
            num_classes,
            output_element_size,
            stream: ctx.default_stream.clone(),
        })
    }

    /// Run recognition inference on a CPU-prepared f16 batch.
    ///
    /// `batch_f16` is [B, 3, H, W] flattened as f16 values.
    /// Returns the output logits/probs as CPU f16 values [B, T, C].
    pub fn infer_batch_cpu(
        &mut self,
        batch_f16: &[f16],
        batch_size: u32,
        actual_width: u32,
    ) -> Result<Vec<f16>> {
        if batch_size > self.max_batch_size {
            return Err(OcrError::Recognition(format!(
                "batch size {batch_size} exceeds max {}",
                self.max_batch_size
            )));
        }

        // Upload input batch to GPU.
        let input_bytes: &[u8] = bytemuck::cast_slice(batch_f16);
        let input_gpu = self
            .stream
            .clone_htod(input_bytes)
            .map_err(|e| OcrError::Cuda(format!("input upload failed: {e}")))?;
        let input_ptr = {
            let (ptr, _sync) = DevicePtr::device_ptr(&input_gpu, &self.stream);
            ptr as u64
        };

        // Bind input tensor and set shape.
        self.context
            .set_tensor_address_raw(&self.input_name, input_ptr)?;
        let w = actual_width.min(self.max_input_width);
        self.context.set_input_shape(
            &self.input_name,
            &[batch_size as i64, 3, self.input_height as i64, w as i64],
        )?;

        // Run inference.
        let stream_ptr = 0u64; // default stream
        self.context.enqueue_v3(stream_ptr)?;

        // Download output.
        let seq_len = w / 4; // SVTRv2 stride = 4
        let output_elements =
            (batch_size * seq_len * self.num_classes) as usize;
        let output_bytes = output_elements * self.output_element_size;

        let output_cpu: Vec<u8> = self
            .stream
            .clone_dtoh(&self.output_slice)
            .map_err(|e| OcrError::Cuda(format!("output download failed: {e}")))?;

        // Reinterpret as f16 (or f32→f16 conversion if needed).
        let output_f16: Vec<f16> = if self.output_element_size == 2 {
            let slice: &[f16] =
                bytemuck::cast_slice(&output_cpu[..output_bytes]);
            slice.to_vec()
        } else {
            // FP32 output: convert to f16.
            let f32_slice: &[f32] =
                bytemuck::cast_slice(&output_cpu[..output_bytes]);
            f32_slice.iter().map(|&v| f16::from_f32(v)).collect()
        };

        Ok(output_f16)
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

    pub fn num_classes(&self) -> u32 {
        self.num_classes
    }
}
