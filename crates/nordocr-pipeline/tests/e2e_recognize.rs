//! End-to-end test: GPU detection + TRT recognition vs C# reference.
//!
//! Runs full pipeline on extracted PNG pages, compares recognized text
//! against the C# .NET Tesseract pipeline's ocr_results.json.

use std::path::Path;

use nordocr_core::{BBox, RawImage, TextRegion};
use nordocr_detect::GpuMorphologicalDetector;
use nordocr_gpu::{GpuContext, GpuContextConfig};
use nordocr_recognize::{CtcDecoder, RecognitionEngine};
use serde::Deserialize;

/// C# pipeline result entry.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct CSharpRegion {
    page_number: u32,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    text: String,
    text_confidence: f64,
    column: i32,
    row: i32,
    density: f64,
    cluster_id: i32,
}

/// Load all page_XX.png files from a directory, sorted by name.
fn load_pages(dir: &Path) -> Vec<RawImage> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .expect("failed to read page directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map_or(false, |ext| ext == "png" || ext == "PNG")
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    entries
        .iter()
        .map(|entry| {
            let img = image::open(entry.path()).expect("failed to load PNG");
            let rgb = img.to_rgb8();
            RawImage {
                width: rgb.width(),
                height: rgb.height(),
                channels: 3,
                data: rgb.into_raw(),
            }
        })
        .collect()
}

/// Crop a text region, resize to target_h preserving aspect ratio,
/// normalize to [-1,1], return as f32 CHW [3, H, W].
fn prepare_line_f32(page: &RawImage, bbox: &BBox, target_h: u32, max_w: u32) -> (Vec<f32>, u32) {
    let pw = page.width as usize;
    let ph = page.height as usize;
    let channels = page.channels as usize;

    let x0 = (bbox.x.max(0.0) as usize).min(pw.saturating_sub(1));
    let y0 = (bbox.y.max(0.0) as usize).min(ph.saturating_sub(1));
    let x1 = ((bbox.x + bbox.width).ceil() as usize).min(pw);
    let y1 = ((bbox.y + bbox.height).ceil() as usize).min(ph);

    let crop_w = x1.saturating_sub(x0).max(1);
    let crop_h = y1.saturating_sub(y0).max(1);

    let scale = target_h as f64 / crop_h as f64;
    let new_w = ((crop_w as f64 * scale).round() as u32).max(4);
    let new_w = (new_w / 4) * 4;
    let new_w = new_w.min(max_w).max(4);
    let new_h = target_h;

    let mut chw = vec![0.0f32; 3 * new_h as usize * new_w as usize];

    for dy in 0..new_h as usize {
        for dx in 0..new_w as usize {
            let sx = dx as f64 * crop_w as f64 / new_w as f64;
            let sy = dy as f64 * crop_h as f64 / new_h as f64;

            let sx0 = (sx as usize).min(crop_w.saturating_sub(1));
            let sy0 = (sy as usize).min(crop_h.saturating_sub(1));
            let sx1 = (sx0 + 1).min(crop_w.saturating_sub(1));
            let sy1 = (sy0 + 1).min(crop_h.saturating_sub(1));

            let fx = sx - sx0 as f64;
            let fy = sy - sy0 as f64;

            for c in 0..3 {
                let src_c = if channels >= 3 { c } else { 0 };
                let get_pixel = |px: usize, py: usize| -> f64 {
                    let idx = (y0 + py) * pw * channels + (x0 + px) * channels + src_c;
                    page.data.get(idx).copied().unwrap_or(128) as f64
                };

                let v00 = get_pixel(sx0, sy0);
                let v10 = get_pixel(sx1, sy0);
                let v01 = get_pixel(sx0, sy1);
                let v11 = get_pixel(sx1, sy1);

                let v = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;

                // Normalize: (v / 255.0 - 0.5) / 0.5 = v / 127.5 - 1.0
                let normalized = (v / 127.5 - 1.0) as f32;
                let dst_idx = c * (new_h as usize * new_w as usize) + dy * new_w as usize + dx;
                chw[dst_idx] = normalized;
            }
        }
    }

    (chw, new_w)
}

#[test]
fn e2e_detect_and_recognize() {
    let pages_dir = Path::new(r"c:\temp\tiff_pages");
    let csharp_json = Path::new(r"C:\Temp\tesseract-pipeline\ocr_results.json");
    let engine_path = Path::new(r"C:\Dev\nordocr\models\recognize_svtrv2_sm86.engine");

    if !pages_dir.exists() || !csharp_json.exists() || !engine_path.exists() {
        eprintln!("Skipping: need pages dir, C# JSON, and TRT engine");
        return;
    }

    // Load C# reference.
    let json_str = std::fs::read_to_string(csharp_json).expect("failed to read JSON");
    let csharp_regions: Vec<CSharpRegion> =
        serde_json::from_str(&json_str).expect("failed to parse JSON");

    let max_page = csharp_regions.iter().map(|r| r.page_number).max().unwrap_or(0);
    let mut csharp_by_page: Vec<Vec<&CSharpRegion>> = vec![vec![]; (max_page + 1) as usize];
    for r in &csharp_regions {
        csharp_by_page[r.page_number as usize].push(r);
    }

    // Load pages.
    let pages = load_pages(pages_dir);
    eprintln!("Loaded {} pages", pages.len());

    // Initialize GPU.
    let gpu = GpuContext::new(GpuContextConfig {
        device_ordinal: 0,
        pool_size: 256 * 1024 * 1024,
        stream_count: 1,
    })
    .expect("failed to init GPU");

    // Initialize detector.
    let detector = GpuMorphologicalDetector::new(&gpu).expect("failed to init GPU detector");

    // Initialize recognition engine.
    let input_h = 32u32;
    let max_w = 1792u32;
    let max_seq = max_w / 4; // SVTRv2 stride = 4
    let max_batch = 64u32;

    let mut engine = RecognitionEngine::load(&gpu, engine_path, max_batch, input_h, max_w, max_seq)
        .expect("failed to load TRT engine");

    let decoder = CtcDecoder::new();

    eprintln!("Engine loaded: input_h={}, max_w={}, max_seq={}, max_batch={}", input_h, max_w, max_seq, max_batch);

    let test_pages = pages.len().min(18);
    let mut total_exact = 0usize;
    let mut total_close = 0usize;
    let mut total_matched = 0usize;
    let mut total_regions = 0usize;

    let detect_start = std::time::Instant::now();
    let mut all_page_regions: Vec<Vec<TextRegion>> = Vec::new();

    for page_idx in 0..test_pages {
        let page = &pages[page_idx];
        let regions = detector
            .detect(&gpu, &page.data, page.width, page.height, page_idx as u32)
            .expect("detection failed");
        all_page_regions.push(regions);
    }
    let detect_elapsed = detect_start.elapsed();
    eprintln!("Detection: {:.1}ms for {} pages", detect_elapsed.as_secs_f64() * 1000.0, test_pages);

    let recog_start = std::time::Instant::now();

    for page_idx in 0..test_pages {
        let page = &pages[page_idx];
        let regions = &all_page_regions[page_idx];

        let cs_page = if page_idx < csharp_by_page.len() {
            &csharp_by_page[page_idx]
        } else {
            &vec![]
        };

        // Prepare all regions for this page.
        let mut prepared: Vec<(BBox, Vec<f32>, u32)> = Vec::new();
        for region in regions {
            let (chw_f32, actual_w) = prepare_line_f32(page, &region.bbox, input_h, max_w);
            prepared.push((region.bbox, chw_f32, actual_w));
        }

        // Sort by width so similar widths batch together (minimizes padding).
        prepared.sort_by_key(|&(_, _, w)| w);

        // Width-bucketed batching: only batch items within max_width_var px of each other.
        // SVTRv2's transformer self-attention doesn't tolerate width padding at all:
        //   0px → 83.3% exact (zero-loss, 1.6x speedup from natural width collisions)
        //  32px → 78.8% exact (padding contaminates attention, leading garbage chars)
        // 100px → 76.2% exact (even worse — heavy padding artifacts on dense pages)
        let max_width_var = 0u32;
        let mut rust_results: Vec<(BBox, String)> = Vec::new();
        let mut chunk_start = 0;

        while chunk_start < prepared.len() {
            let base_w = prepared[chunk_start].2;
            let mut chunk_end = chunk_start + 1;
            while chunk_end < prepared.len()
                && chunk_end - chunk_start < max_batch as usize
                && prepared[chunk_end].2 - base_w <= max_width_var
            {
                chunk_end += 1;
            }
            let chunk = &prepared[chunk_start..chunk_end];
            chunk_start = chunk_end;
            let batch_size = chunk.len() as u32;
            let batch_w = chunk.iter().map(|(_, _, w)| *w).max().unwrap_or(4);
            let pixels_per_item = 3 * input_h as usize * batch_w as usize;

            // Pad with 1.0 (white background in [-1,1] normalized space).
            // 0.0 = gray, which the model interprets as content and generates garbage.
            let mut batch_data = vec![1.0f32; batch_size as usize * pixels_per_item];
            for (i, (_, chw, item_w)) in chunk.iter().enumerate() {
                let src_row = *item_w as usize;
                let dst_row = batch_w as usize;
                // Copy channel by channel, row by row (CHW layout).
                for c in 0..3usize {
                    for y in 0..input_h as usize {
                        let src_offset = c * (input_h as usize * src_row) + y * src_row;
                        let dst_offset =
                            i * pixels_per_item + c * (input_h as usize * dst_row) + y * dst_row;
                        let copy_len = src_row.min(dst_row);
                        batch_data[dst_offset..dst_offset + copy_len]
                            .copy_from_slice(&chw[src_offset..src_offset + copy_len]);
                        // Remaining columns already 1.0 (right-pad with white).
                    }
                }
            }

            let output_f16 = engine
                .infer_batch_f32(&batch_data, batch_size, batch_w)
                .expect("inference failed");

            // Decode each item with its own seq_len to avoid garbage from padding columns.
            // Output layout: [batch, batch_seq, 126] where batch_seq = batch_w / 4.
            let batch_seq = (batch_w / 4) as usize;
            let output_item_stride = batch_seq * 126;

            for (i, (bbox, _, item_w)) in chunk.iter().enumerate() {
                let item_seq = *item_w / 4;
                let out_start = i * output_item_stride;
                let out_end = out_start + item_seq as usize * 126;
                let item_output = &output_f16[out_start..out_end];
                let decoded = decoder
                    .decode_cpu(item_output, 1, item_seq)
                    .expect("decode failed");
                rust_results.push((*bbox, decoded[0].text.clone()));
            }
        }

        // Match against C# by center proximity and compare text.
        let mut page_exact = 0;
        let mut page_close = 0;
        let mut page_matched = 0;

        let mut cs_used = vec![false; cs_page.len()];

        for (rust_bbox, rust_text) in &rust_results {
            let rust_cx = rust_bbox.x + rust_bbox.width / 2.0;
            let rust_cy = rust_bbox.y + rust_bbox.height / 2.0;

            // Find closest unmatched C# region.
            let mut best_ci = None;
            let mut best_dist = f32::MAX;
            for (ci, cs) in cs_page.iter().enumerate() {
                if cs_used[ci] {
                    continue;
                }
                let cs_cx = cs.x as f32 + cs.width as f32 / 2.0;
                let cs_cy = cs.y as f32 + cs.height as f32 / 2.0;
                let dist = ((rust_cx - cs_cx).powi(2) + (rust_cy - cs_cy).powi(2)).sqrt();
                if dist < 100.0 && dist < best_dist {
                    best_dist = dist;
                    best_ci = Some(ci);
                }
            }

            if let Some(ci) = best_ci {
                cs_used[ci] = true;
                page_matched += 1;

                let cs_text = &cs_page[ci].text;
                let rust_text_trimmed = rust_text.trim();
                let cs_text_trimmed = cs_text.trim();

                if rust_text_trimmed == cs_text_trimmed {
                    page_exact += 1;
                } else {
                    // Check case-insensitive match.
                    let close = rust_text_trimmed.to_lowercase() == cs_text_trimmed.to_lowercase();
                    if close {
                        page_close += 1;
                    }

                    // Print differences for first few pages.
                    if page_idx < 5 || !close {
                        let display_len = 60;
                        let cs_short: String = cs_text_trimmed.chars().take(display_len).collect();
                        let ru_short: String = rust_text_trimmed.chars().take(display_len).collect();
                        eprintln!(
                            "  p{} row{}: C#=\"{}\" Rust=\"{}\"{}",
                            page_idx,
                            cs_page[ci].row,
                            cs_short,
                            ru_short,
                            if close { " [case]" } else { "" }
                        );
                    }
                }
            }
        }

        eprintln!(
            "Page {}: {} regions, {} matched, {} exact, {} case-match, {} different",
            page_idx,
            rust_results.len(),
            page_matched,
            page_exact,
            page_close,
            page_matched - page_exact - page_close,
        );

        total_exact += page_exact;
        total_close += page_close;
        total_matched += page_matched;
        total_regions += rust_results.len();
    }

    let recog_elapsed = recog_start.elapsed();

    eprintln!("\n=== RECOGNITION SUMMARY ({} pages) ===", test_pages);
    eprintln!("Total regions: {}", total_regions);
    eprintln!(
        "Detection: {:.1}ms, Recognition: {:.1}ms, Total: {:.1}ms",
        detect_elapsed.as_secs_f64() * 1000.0,
        recog_elapsed.as_secs_f64() * 1000.0,
        (detect_elapsed + recog_elapsed).as_secs_f64() * 1000.0,
    );
    eprintln!("Matched with C#: {}", total_matched);
    eprintln!(
        "Exact text match: {} ({:.1}%)",
        total_exact,
        total_exact as f64 / total_matched.max(1) as f64 * 100.0
    );
    eprintln!(
        "Case-insensitive match: {} ({:.1}%)",
        total_close,
        total_close as f64 / total_matched.max(1) as f64 * 100.0
    );
    eprintln!(
        "Different: {} ({:.1}%)",
        total_matched - total_exact - total_close,
        (total_matched - total_exact - total_close) as f64 / total_matched.max(1) as f64 * 100.0
    );
}
