use serde::{Deserialize, Serialize};

/// Runtime configuration for the OCR pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Path to the detection TensorRT engine file.
    pub detect_engine_path: String,
    /// Path to the recognition TensorRT engine file.
    pub recognize_engine_path: String,

    // Detection settings.
    /// Maximum batch size for detection (pages per batch).
    pub detect_max_batch: u32,
    /// Detection model input height.
    pub detect_input_height: u32,
    /// Detection model input width.
    pub detect_input_width: u32,
    /// Probability threshold for text detection.
    pub detect_threshold: f32,
    /// Minimum area for detected text regions.
    pub detect_min_area: f32,

    // Recognition settings.
    /// Maximum batch size for recognition (text lines per batch).
    pub recognize_max_batch: u32,
    /// Recognition model input height.
    pub recognize_input_height: u32,
    /// Maximum recognition model input width.
    pub recognize_max_input_width: u32,
    /// Maximum output sequence length.
    pub recognize_max_seq_len: u32,

    // Pipeline settings.
    /// Number of CUDA streams for page-level parallelism.
    pub num_streams: usize,
    /// GPU memory pool initial size in bytes.
    pub gpu_pool_size: usize,
    /// Whether to capture the pipeline as a CUDA graph.
    pub enable_cuda_graph: bool,
    /// Whether to run preprocessing.
    pub enable_preprocess: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            detect_engine_path: "models/detect.engine".to_string(),
            recognize_engine_path: "models/recognize.engine".to_string(),

            detect_max_batch: 4,
            detect_input_height: 1024,
            detect_input_width: 1024,
            detect_threshold: 0.3,
            detect_min_area: 100.0,

            recognize_max_batch: 64,
            recognize_input_height: 32,
            recognize_max_input_width: 512,
            recognize_max_seq_len: 64,

            num_streams: 4,
            gpu_pool_size: 256 * 1024 * 1024, // 256 MB
            enable_cuda_graph: true,
            enable_preprocess: true,
        }
    }
}
