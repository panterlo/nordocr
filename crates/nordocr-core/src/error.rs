use thiserror::Error;

/// Top-level error type for the nordocr pipeline.
#[derive(Debug, Error)]
pub enum OcrError {
    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("TensorRT error: {0}")]
    TensorRt(String),

    #[error("Image decode error: {0}")]
    ImageDecode(String),

    #[error("PDF rendering error: {0}")]
    PdfRender(String),

    #[error("Model loading error: {0}")]
    ModelLoad(String),

    #[error("Preprocessing error: {0}")]
    Preprocess(String),

    #[error("Detection error: {0}")]
    Detection(String),

    #[error("Recognition error: {0}")]
    Recognition(String),

    #[error("Pipeline error: {0}")]
    Pipeline(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("GPU memory allocation failed: requested {requested} bytes, available {available} bytes")]
    GpuOutOfMemory { requested: usize, available: usize },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, OcrError>;
