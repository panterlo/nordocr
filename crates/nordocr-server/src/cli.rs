use std::path::PathBuf;

use clap::{Parser, Subcommand};

use nordocr_core::{FileInput, Result};
use nordocr_pipeline::{OcrPipeline, PipelineConfig};

#[derive(Parser)]
#[command(name = "nordocr", about = "Nordic OCR Engine — GPU-accelerated document OCR")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    /// GPU device ordinal.
    #[arg(long, default_value = "0", global = true)]
    pub device: usize,

    /// Path to pipeline config file (TOML/JSON).
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    /// Log level (trace, debug, info, warn, error).
    #[arg(long, default_value = "info", global = true)]
    pub log_level: String,
}

#[derive(Subcommand)]
pub enum Command {
    /// Start the HTTP server.
    Serve {
        /// Host to bind to.
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        /// Port to bind to.
        #[arg(long, default_value = "8080")]
        port: u16,
    },
    /// Process a single file and print results.
    Process {
        /// Input file path (image or PDF).
        #[arg(required = true)]
        input: PathBuf,
        /// Output format.
        #[arg(long, default_value = "text")]
        format: OutputFormat,
        /// Pages to process (for PDFs, comma-separated).
        #[arg(long)]
        pages: Option<String>,
    },
    /// Process a directory of files in batch.
    Batch {
        /// Input directory.
        #[arg(required = true)]
        input_dir: PathBuf,
        /// Output directory for results.
        #[arg(required = true)]
        output_dir: PathBuf,
        /// Output format.
        #[arg(long, default_value = "json")]
        format: OutputFormat,
        /// Number of worker threads.
        #[arg(long, default_value = "4")]
        workers: usize,
    },
    /// Build TensorRT engines from ONNX models.
    BuildEngines {
        /// Path to detection ONNX model.
        #[arg(long)]
        detect_onnx: PathBuf,
        /// Path to recognition ONNX model.
        #[arg(long)]
        recognize_onnx: PathBuf,
        /// Output directory for engine files.
        #[arg(long, default_value = "models")]
        output_dir: PathBuf,
    },
}

#[derive(Clone, Debug, clap::ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
    Jsonl,
}

/// Process a single file via CLI.
pub fn process_file(
    pipeline: &mut OcrPipeline,
    path: &PathBuf,
    format: &OutputFormat,
    pages: Option<&[u32]>,
) -> Result<()> {
    let data = std::fs::read(path)?;
    let input = FileInput::from_bytes(data);

    let (results, timing) = pipeline.process(&input, pages)?;

    match format {
        OutputFormat::Text => {
            for page in &results {
                if results.len() > 1 {
                    println!("--- Page {} ---", page.page_index + 1);
                }
                println!("{}", page.text);
            }
            eprintln!(
                "\n[{:.1}ms total | decode:{:.1}ms preproc:{:.1}ms detect:{:.1}ms recog:{:.1}ms]",
                timing.total_ms,
                timing.decode_ms,
                timing.preprocess_ms,
                timing.detect_ms,
                timing.recognize_ms
            );
        }
        OutputFormat::Json => {
            let output = serde_json::json!({
                "pages": results,
                "timing": timing,
            });
            println!("{}", serde_json::to_string_pretty(&output).unwrap());
        }
        OutputFormat::Jsonl => {
            for page in &results {
                println!("{}", serde_json::to_string(page).unwrap());
            }
        }
    }

    Ok(())
}
