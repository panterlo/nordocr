use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::Json;
use axum::routing::{get, post};
use axum::Router;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use base64::Engine as _;
use nordocr_core::{FileInput, PageResult, TimingInfo};
use nordocr_pipeline::OcrPipeline;

/// Shared application state.
pub struct AppState {
    pub pipeline: Mutex<OcrPipeline>,
    pub start_time: Instant,
}

/// POST /ocr request body (JSON variant).
#[derive(Deserialize)]
pub struct OcrJsonRequest {
    /// Base64-encoded file data.
    pub file: String,
    /// Pages to process (for PDFs). None = all.
    pub pages: Option<Vec<u32>>,
    /// Override language (default: auto-detect Nordic).
    pub language: Option<String>,
    /// Return word-level bounding boxes.
    pub word_boxes: Option<bool>,
}

/// POST /ocr response.
#[derive(Serialize)]
pub struct OcrResponse {
    pub pages: Vec<PageResult>,
    pub timing: TimingInfo,
}

/// GET /health response.
#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub uptime_secs: f64,
    pub gpu_ready: bool,
}

/// Error response body.
#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/ocr", post(ocr_multipart))
        .route("/ocr/json", post(ocr_json))
        .route("/health", get(health))
        .with_state(state)
}

/// POST /ocr — accepts multipart/form-data with a file field.
async fn ocr_multipart(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<OcrResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mut file_data: Option<Vec<u8>> = None;
    let mut pages: Option<Vec<u32>> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| bad_request(format!("multipart error: {e}")))?
    {
        match field.name() {
            Some("file") => {
                file_data = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| bad_request(format!("file read error: {e}")))?
                        .to_vec(),
                );
            }
            Some("pages") => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| bad_request(format!("pages field error: {e}")))?;
                pages = serde_json::from_str(&text)
                    .map_err(|e| bad_request(format!("invalid pages: {e}")))?;
            }
            _ => {}
        }
    }

    let data = file_data.ok_or_else(|| bad_request("missing 'file' field".into()))?;
    let input = FileInput::from_bytes(data);

    process_ocr(&state, input, pages.as_deref())
}

/// POST /ocr/json — accepts JSON with base64-encoded file.
async fn ocr_json(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OcrJsonRequest>,
) -> Result<Json<OcrResponse>, (StatusCode, Json<ErrorResponse>)> {
    let data = base64::engine::general_purpose::STANDARD
        .decode(&req.file)
        .map_err(|e| bad_request(format!("invalid base64: {e}")))?;

    let input = FileInput::from_bytes(data);

    process_ocr(&state, input, req.pages.as_deref())
}

fn process_ocr(
    state: &AppState,
    input: FileInput,
    pages: Option<&[u32]>,
) -> Result<Json<OcrResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mut pipeline = state.pipeline.lock();

    let (page_results, timing) = pipeline
        .process(&input, pages)
        .map_err(|e| internal_error(format!("pipeline error: {e}")))?;

    metrics::counter!("ocr_requests_total").increment(1);
    metrics::histogram!("ocr_latency_ms").record(timing.total_ms as f64);

    Ok(Json(OcrResponse {
        pages: page_results,
        timing,
    }))
}

/// GET /health — liveness + readiness check.
async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        uptime_secs: state.start_time.elapsed().as_secs_f64(),
        gpu_ready: true, // In production: check GPU context validity.
    })
}

fn bad_request(msg: String) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse { error: msg }),
    )
}

fn internal_error(msg: String) -> (StatusCode, Json<ErrorResponse>) {
    tracing::error!(error = %msg, "internal error");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse { error: msg }),
    )
}
