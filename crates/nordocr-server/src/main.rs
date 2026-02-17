use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use tokio::signal;
use tracing_subscriber::EnvFilter;

use nordocr_pipeline::{OcrPipeline, PipelineConfig};

mod api;
mod cli;

use cli::{Cli, Command};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize tracing.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(&cli.log_level)),
        )
        .with_target(true)
        .with_thread_ids(true)
        .init();

    tracing::info!("nordocr v{}", env!("CARGO_PKG_VERSION"));

    // Load or create config.
    let mut config = if let Some(config_path) = &cli.config {
        let data = std::fs::read_to_string(config_path)?;
        serde_json::from_str(&data)?
    } else {
        PipelineConfig::default()
    };

    // Apply CLI overrides for recognition backend.
    if let Some(backend) = &cli.recognize {
        match backend {
            cli::RecognizeBackend::Svtrv2 => {
                config.recognize_model = nordocr_pipeline::RecogModelArch::SVTRv2;
            }
            cli::RecognizeBackend::Tesseract => {
                config.recognize_model = nordocr_pipeline::RecogModelArch::Tesseract;
            }
        }
    }
    if let Some(dll) = &cli.tesseract_dll {
        config.tesseract_dll_path = Some(dll.to_string_lossy().into_owned());
    }
    if let Some(td) = &cli.tessdata {
        config.tessdata_path = Some(td.to_string_lossy().into_owned());
    }
    if let Some(lang) = &cli.tess_lang {
        config.tess_language = Some(lang.clone());
    }

    match cli.command {
        Command::Serve { host, port } => {
            serve(config, &host, port).await?;
        }
        Command::Process {
            input,
            format,
            pages,
        } => {
            let mut pipeline = OcrPipeline::build(config)?;
            let page_nums: Option<Vec<u32>> = pages.map(|s| {
                s.split(',')
                    .filter_map(|p| p.trim().parse().ok())
                    .collect()
            });
            cli::process_file(
                &mut pipeline,
                &input,
                &format,
                page_nums.as_deref(),
            )?;
        }
        Command::Batch {
            input_dir,
            output_dir,
            format,
            workers,
        } => {
            tracing::info!(
                input = %input_dir.display(),
                output = %output_dir.display(),
                workers,
                "batch processing"
            );
            // In production: scan directory, distribute files across workers.
            let mut pipeline = OcrPipeline::build(config)?;
            std::fs::create_dir_all(&output_dir)?;

            for entry in std::fs::read_dir(&input_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    tracing::info!(file = %path.display(), "processing");
                    if let Err(e) = cli::process_file(&mut pipeline, &path, &format, None) {
                        tracing::error!(file = %path.display(), error = %e, "failed");
                    }
                }
            }
        }
        Command::BuildEngines {
            detect_onnx,
            recognize_onnx,
            output_dir,
        } => {
            use nordocr_trt::TrtEngineBuilder;

            std::fs::create_dir_all(&output_dir)?;

            tracing::info!("building detection engine");
            TrtEngineBuilder::new()
                .with_fp8()
                .with_sparsity()
                .max_batch_size(config.detect_max_batch)
                .build_from_onnx(
                    &detect_onnx,
                    &output_dir.join("detect.engine"),
                    None,
                )?;

            tracing::info!("building recognition engine");
            TrtEngineBuilder::new()
                .with_fp8()
                .max_batch_size(config.recognize_max_batch)
                .build_from_onnx(
                    &recognize_onnx,
                    &output_dir.join("recognize.engine"),
                    None,
                )?;

            tracing::info!("engines built successfully");
        }
    }

    Ok(())
}

async fn serve(config: PipelineConfig, host: &str, port: u16) -> anyhow::Result<()> {
    // Initialize Prometheus metrics exporter.
    let metrics_builder = metrics_exporter_prometheus::PrometheusBuilder::new();
    let metrics_handle = metrics_builder
        .install_recorder()
        .expect("failed to install metrics recorder");

    // Build pipeline.
    let mut pipeline = OcrPipeline::build(config)?;

    // Warm up.
    pipeline.warmup()?;

    let state = Arc::new(api::AppState {
        pipeline: parking_lot::Mutex::new(pipeline),
        start_time: Instant::now(),
    });

    // Build router with middleware.
    let app = api::create_router(state)
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .layer(tower_http::cors::CorsLayer::permissive());

    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    tracing::info!(%addr, "starting HTTP server");

    let listener = tokio::net::TcpListener::bind(addr).await?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("server stopped");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c().await.expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("shutdown signal received");
}
