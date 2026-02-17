use nordocr_core::{BBox, Polygon, TextRegion};

use crate::contour;

/// CPU-based morphological text region detector.
///
/// Algorithm (ported from Ormeo.Document `TextRegionOcrStep.cs`):
/// 1. RGB → grayscale (BT.601 luma)
/// 2. Gaussian blur (5×5, noise reduction)
/// 3. Adaptive threshold + invert (dark text → white foreground)
/// 4. Separable binary dilation with rect kernel (connects chars into lines)
/// 5. Connected component labeling → bounding boxes with orientation
/// 6. Filter by size, margin (with 50% overlap tolerance), and rotation angle
/// 7. Sort top-to-bottom, left-to-right
/// 8. Row/column clustering and cluster (table section) detection
pub struct MorphologicalDetector {
    // --- Dilation settings (unchanged from C#) ---
    /// Dilation kernel width (connects characters horizontally). Default: 20.
    pub kernel_w: u32,
    /// Dilation kernel height (keeps text lines separate). Default: 3.
    pub kernel_h: u32,
    /// Number of dilation iterations. Default: 2.
    pub iterations: u32,

    // --- Blur settings ---
    /// Gaussian blur kernel size (must be odd, 0 = disabled). Default: 5.
    pub blur_size: u32,

    // --- Adaptive threshold settings ---
    /// Block size for adaptive thresholding (must be odd). Default: 15.
    pub adaptive_block_size: u32,
    /// Constant subtracted from local mean in adaptive threshold. Default: 4.0.
    pub adaptive_c: f64,

    // --- Size filter ---
    /// Minimum bounding box width to keep. Default: 20.
    pub min_width: u32,
    /// Minimum bounding box height to keep. Default: 24.
    pub min_height: u32,

    // --- Margin filter (asymmetric, with 50% overlap tolerance) ---
    /// Margin in pixels at left and right page edges. Default: 140.
    pub side_margin: u32,
    /// Margin in pixels at the top page edge. Default: 120.
    pub top_margin: u32,
    /// Margin in pixels at the bottom page edge. Default: 140.
    pub bottom_margin: u32,

    // --- Rotation filter ---
    /// Whether to filter regions with non-horizontal/vertical orientation. Default: true.
    pub filter_rotation: bool,
    /// Minimum angle (degrees) to consider a region as rotated. Default: 4.0.
    pub rotation_angle_min: f32,
    /// Maximum angle (degrees) for the "rotated" range. Default: 86.0.
    /// Regions with orientation in [min, max] or [180-max, 180-min] are filtered.
    pub rotation_angle_max: f32,

    // --- Multi-line splitting ---
    /// Maximum component height as ratio of median height before splitting.
    /// Components taller than `median_height * max_height_ratio` are split at
    /// horizontal projection valleys. Default: 1.8.
    pub max_height_ratio: f32,

    // --- Padding ---
    /// Padding added around each detected region (pixels). Default: 8.
    pub region_padding: u32,

    // --- Row/column clustering ---
    /// Maximum Y-center distance (pixels) for regions to be in the same row. Default: 20.
    pub max_row_y_distance: u32,
    /// Minimum vertical gap (pixels) between rows to start a new cluster. Default: 80.
    pub min_cluster_gap: u32,
    /// Minimum region height to include in row/column clustering. Default: 20.
    pub cluster_min_height: u32,
}

impl Default for MorphologicalDetector {
    /// Defaults from the Ormeo.Document C# pipeline (`CreateDefaultOcrSettings` +
    /// `CreateDefaultArtifactSettings`).
    fn default() -> Self {
        Self {
            kernel_w: 20,
            kernel_h: 3,
            iterations: 2,
            blur_size: 5,
            adaptive_block_size: 15,
            adaptive_c: 4.0,
            min_width: 20,
            min_height: 24,
            side_margin: 140,
            top_margin: 120,
            bottom_margin: 140,
            filter_rotation: true,
            rotation_angle_min: 4.0,
            rotation_angle_max: 86.0,
            max_height_ratio: 1.8,
            region_padding: 8,
            max_row_y_distance: 20,
            min_cluster_gap: 80,
            cluster_min_height: 20,
        }
    }
}

impl MorphologicalDetector {
    /// Detect text regions in an RGB u8 HWC image (CPU memory).
    ///
    /// Returns text regions sorted top-to-bottom, left-to-right,
    /// with row/column/cluster assignments.
    pub fn detect(
        &self,
        rgb_image: &[u8],
        width: u32,
        height: u32,
        page_index: u32,
    ) -> Vec<TextRegion> {
        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        // Step 1: RGB → grayscale (BT.601 luma).
        let mut gray = vec![0u8; n];
        for i in 0..n {
            let r = rgb_image[i * 3] as u32;
            let g = rgb_image[i * 3 + 1] as u32;
            let b = rgb_image[i * 3 + 2] as u32;
            gray[i] = ((r * 77 + g * 150 + b * 29) >> 8) as u8;
        }

        // Step 2: Gaussian blur (noise reduction before thresholding).
        if self.blur_size >= 3 {
            let mut blurred = vec![0u8; n];
            gaussian_blur(&gray, &mut blurred, w, h, self.blur_size as usize);
            gray = blurred;
        }

        // Step 3: Adaptive threshold + invert (dark text → white foreground).
        let mut binary = vec![0u8; n];
        adaptive_threshold_mean(
            &gray,
            &mut binary,
            w,
            h,
            self.adaptive_block_size as usize,
            self.adaptive_c,
        );
        drop(gray);

        // Step 4: Separable binary dilation.
        // Keep pre-dilation binary for edge fragment trimming.
        let pre_dilation = binary.clone();
        let mut current = binary;
        for _ in 0..self.iterations {
            let mut h_dilated = vec![0u8; n];
            dilate_horizontal(&current, &mut h_dilated, w, h, self.kernel_w as usize);
            let mut v_dilated = vec![0u8; n];
            dilate_vertical(&h_dilated, &mut v_dilated, w, h, self.kernel_h as usize);
            current = v_dilated;
        }

        // Step 5: CCL → bounding boxes with orientation.
        let components = contour::extract_components(&current, width, height);

        // Step 5b: Split multi-line merges (vertical).
        let components =
            contour::split_tall_components(components, &current, width, height, self.max_height_ratio);
        drop(current);

        // Step 5c: Trim small edge fragments (removes leading/trailing garbage from
        // adjacent text elements that were bridged by dilation).
        let components = contour::trim_edge_fragments(components, &pre_dilation, width, height);
        drop(pre_dilation);

        tracing::debug!(
            page = page_index,
            raw_regions = components.len(),
            "morphological detection: raw components"
        );

        // Steps 6-8: Filter, sort, and cluster (shared with GPU path).
        self.filter_and_cluster(components, width, height, page_index)
    }

    /// Filter components by size/margin/rotation, sort, pad, and assign row/column clusters.
    ///
    /// This is the CPU post-processing shared between the CPU and GPU detection paths.
    /// The GPU path produces identical components via GPU kernels + CPU CCL, then
    /// calls this method for filtering and clustering.
    pub fn filter_and_cluster(
        &self,
        components: Vec<contour::ComponentInfo>,
        width: u32,
        height: u32,
        page_index: u32,
    ) -> Vec<TextRegion> {
        let min_w = self.min_width as f32;
        let min_h = self.min_height as f32;
        let page_w = width as f32;
        let page_h = height as f32;

        let mut regions: Vec<TextRegion> = components
            .into_iter()
            .filter(|c| {
                // Size filter
                if c.bbox.width < min_w || c.bbox.height < min_h {
                    return false;
                }

                // Margin filter with 50% overlap tolerance (matches C#)
                if !self.passes_margin_filter(&c.bbox, page_w, page_h) {
                    return false;
                }

                // Rotation filter
                if self.filter_rotation {
                    let angle = c.orientation_deg;
                    let in_range = |a: f32| {
                        a > self.rotation_angle_min && a < self.rotation_angle_max
                    };
                    if in_range(angle) || in_range(180.0 - angle) {
                        return false;
                    }
                }

                true
            })
            .map(|c| {
                let pad = self.region_padding as f32;
                let x = (c.bbox.x - pad).max(0.0);
                let y = (c.bbox.y - pad).max(0.0);
                let w = (c.bbox.width + 2.0 * pad).min(page_w - x);
                let h = (c.bbox.height + 2.0 * pad).min(page_h - y);
                let bbox = BBox::new(x, y, w, h);
                TextRegion {
                    polygon: Polygon {
                        points: vec![
                            (x, y),
                            (x + w, y),
                            (x + w, y + h),
                            (x, y + h),
                        ],
                    },
                    bbox,
                    confidence: 1.0,
                    page_index,
                    row: None,
                    column: None,
                    cluster_id: None,
                }
            })
            .collect();

        // Sort top-to-bottom, left-to-right.
        regions.sort_by(|a, b| {
            let y_cmp = a.bbox.y.partial_cmp(&b.bbox.y).unwrap();
            if y_cmp == std::cmp::Ordering::Equal {
                a.bbox.x.partial_cmp(&b.bbox.x).unwrap()
            } else {
                y_cmp
            }
        });

        tracing::debug!(
            page = page_index,
            filtered_regions = regions.len(),
            "morphological detection: after filtering"
        );

        // Row/column clustering.
        self.assign_rows_and_columns(&mut regions);

        regions
    }

    /// Margin filter with 50% overlap tolerance, matching the C# implementation.
    ///
    /// If a region starts inside a margin zone, it's only kept if >50% of its
    /// extent reaches past the margin. Regions entirely in the far margin are
    /// always filtered.
    pub fn passes_margin_filter(&self, bbox: &BBox, page_w: f32, page_h: f32) -> bool {
        let left = bbox.x;
        let top = bbox.y;
        let right = bbox.x + bbox.width;
        let bottom = bbox.y + bbox.height;

        let side_m = self.side_margin as f32;
        let top_m = self.top_margin as f32;
        let bottom_m = self.bottom_margin as f32;

        // Left margin check
        if side_m > 0.0 && left < side_m {
            let overflow = right - side_m;
            if overflow < 0.0 || overflow < 0.5 * bbox.width {
                return false;
            }
        }

        // Right margin check — region starts entirely past the right margin
        if side_m > 0.0 {
            let right_margin_x = page_w - side_m;
            if left > right_margin_x {
                return false;
            }
        }

        // Top margin check
        if top_m > 0.0 && top < top_m {
            let overflow = bottom - top_m;
            if overflow < 0.0 || overflow < 0.5 * bbox.height {
                return false;
            }
        }

        // Bottom margin check — region starts entirely past the bottom margin
        if bottom_m > 0.0 {
            let bottom_margin_y = page_h - bottom_m;
            if top > bottom_margin_y {
                return false;
            }
        }

        true
    }

    /// Assign row, column, and cluster IDs to detected regions.
    ///
    /// Matches the C# `RunAgglomerativeClustering` logic:
    /// 1. Group into rows by Y-center proximity
    /// 2. Assign column indices within each row (by X order)
    /// 3. Detect table sections (clusters) by large vertical gaps between rows
    pub fn assign_rows_and_columns(&self, regions: &mut [TextRegion]) {
        let min_h = self.cluster_min_height as f32;

        // Collect indices of regions tall enough for clustering.
        let mut eligible: Vec<usize> = regions
            .iter()
            .enumerate()
            .filter(|(_, r)| r.bbox.height >= min_h)
            .map(|(i, _)| i)
            .collect();

        if eligible.is_empty() {
            return;
        }

        // Sort eligible indices by Y-center.
        eligible.sort_by(|&a, &b| {
            let ya = regions[a].bbox.center().1;
            let yb = regions[b].bbox.center().1;
            ya.partial_cmp(&yb).unwrap()
        });

        // Group into rows by Y-center proximity.
        let max_y_dist = self.max_row_y_distance as f32;
        let mut rows: Vec<Vec<usize>> = Vec::new();
        let mut current_row: Vec<usize> = vec![eligible[0]];

        for &idx in &eligible[1..] {
            let y_center = regions[idx].bbox.center().1;
            let row_avg_y: f32 = current_row
                .iter()
                .map(|&i| regions[i].bbox.center().1)
                .sum::<f32>()
                / current_row.len() as f32;

            if (y_center - row_avg_y).abs() <= max_y_dist {
                current_row.push(idx);
            } else {
                rows.push(current_row);
                current_row = vec![idx];
            }
        }
        rows.push(current_row);

        // Assign row and column indices.
        for (row_idx, row) in rows.iter_mut().enumerate() {
            // Sort by X within the row.
            row.sort_by(|&a, &b| {
                regions[a]
                    .bbox
                    .x
                    .partial_cmp(&regions[b].bbox.x)
                    .unwrap()
            });

            for (col_idx, &region_idx) in row.iter().enumerate() {
                regions[region_idx].row = Some(row_idx as u32);
                regions[region_idx].column = Some(col_idx as u32);
            }
        }

        // Detect clusters by vertical gaps between rows.
        if !rows.is_empty() {
            let min_gap = self.min_cluster_gap as f32;
            let mut cluster_id: u32 = 0;

            // Assign cluster 0 to first row.
            for &idx in &rows[0] {
                regions[idx].cluster_id = Some(cluster_id);
            }

            for i in 1..rows.len() {
                let prev_bottom = rows[i - 1]
                    .iter()
                    .map(|&idx| regions[idx].bbox.bottom())
                    .fold(f32::MIN, f32::max);
                let curr_top = rows[i]
                    .iter()
                    .map(|&idx| regions[idx].bbox.y)
                    .fold(f32::MAX, f32::min);
                let gap = curr_top - prev_bottom;

                if gap > min_gap {
                    cluster_id += 1;
                }

                for &idx in &rows[i] {
                    regions[idx].cluster_id = Some(cluster_id);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Image processing helpers
// ---------------------------------------------------------------------------

/// Separable Gaussian blur using binomial kernel [1, 4, 6, 4, 1] / 16.
///
/// For kernel sizes > 5, the 5×5 kernel is applied, matching the C# pipeline's
/// default GaussianBlur(kSize=5). Edges use clamped (replicate) boundary.
fn gaussian_blur(input: &[u8], output: &mut [u8], w: usize, h: usize, _kernel_size: usize) {
    // Binomial 5-tap kernel ≈ Gaussian with σ ≈ 1.1 (OpenCV default for ksize=5).
    const WEIGHTS: [u32; 5] = [1, 4, 6, 4, 1];
    const SUM: u32 = 16;

    let mut temp = vec![0u8; w * h];

    // Horizontal pass.
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0u32;
            for k in 0..5usize {
                let sx = (x as isize + k as isize - 2).clamp(0, w as isize - 1) as usize;
                acc += input[y * w + sx] as u32 * WEIGHTS[k];
            }
            temp[y * w + x] = (acc / SUM) as u8;
        }
    }

    // Vertical pass.
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0u32;
            for k in 0..5usize {
                let sy = (y as isize + k as isize - 2).clamp(0, h as isize - 1) as usize;
                acc += temp[sy * w + x] as u32 * WEIGHTS[k];
            }
            output[y * w + x] = (acc / SUM) as u8;
        }
    }
}

/// Adaptive mean thresholding using integral images.
///
/// Matches OpenCV `adaptiveThreshold(ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV)`:
/// for each pixel, computes the local mean over a `block_size × block_size` window,
/// then sets foreground (255) if the pixel intensity is below `(mean - c)`.
fn adaptive_threshold_mean(
    gray: &[u8],
    output: &mut [u8],
    w: usize,
    h: usize,
    block_size: usize,
    c: f64,
) {
    // Integral image with +1 border for simpler boundary handling.
    // integral[y+1][x+1] = sum of gray[0..y][0..x]
    let iw = w + 1;
    let mut integral = vec![0i64; iw * (h + 1)];

    for y in 0..h {
        let mut row_sum = 0i64;
        for x in 0..w {
            row_sum += gray[y * w + x] as i64;
            integral[(y + 1) * iw + (x + 1)] = row_sum + integral[y * iw + (x + 1)];
        }
    }

    let half = (block_size / 2) as isize;

    for y in 0..h {
        for x in 0..w {
            // Window bounds clamped to image.
            let y0 = (y as isize - half).max(0) as usize;
            let x0 = (x as isize - half).max(0) as usize;
            let y1 = ((y as isize + half).min(h as isize - 1) + 1) as usize;
            let x1 = ((x as isize + half).min(w as isize - 1) + 1) as usize;

            let area = ((y1 - y0) * (x1 - x0)) as f64;
            let sum = integral[y1 * iw + x1] as f64
                - integral[y0 * iw + x1] as f64
                - integral[y1 * iw + x0] as f64
                + integral[y0 * iw + x0] as f64;

            let mean = sum / area;
            let thresh = mean - c;

            // BINARY_INV: dark text (below threshold) → white foreground.
            output[y * w + x] = if (gray[y * w + x] as f64) <= thresh {
                255
            } else {
                0
            };
        }
    }
}

/// Separable horizontal binary dilation (sliding window maximum).
///
/// For a kernel of width `kernel_w`, each output pixel is 255 if any
/// input pixel in the horizontal window `[x - r_left, x + r_right]` is > 0.
fn dilate_horizontal(input: &[u8], output: &mut [u8], w: usize, h: usize, kernel_w: usize) {
    if kernel_w == 0 {
        output.copy_from_slice(input);
        return;
    }
    let r_left = (kernel_w - 1) / 2;
    let r_right = kernel_w / 2;

    for y in 0..h {
        let row = y * w;
        let mut count = 0usize;

        // Initialize window covering [0, min(r_right, w-1)].
        for x in 0..=r_right.min(w - 1) {
            if input[row + x] > 0 {
                count += 1;
            }
        }

        for x in 0..w {
            output[row + x] = if count > 0 { 255 } else { 0 };

            // Pixel entering right edge of next window.
            let enter = x + r_right + 1;
            if enter < w && input[row + enter] > 0 {
                count += 1;
            }

            // Pixel leaving left edge of current window.
            if x >= r_left && input[row + x - r_left] > 0 {
                count -= 1;
            }
        }
    }
}

/// Separable vertical binary dilation (sliding window maximum).
fn dilate_vertical(input: &[u8], output: &mut [u8], w: usize, h: usize, kernel_h: usize) {
    if kernel_h == 0 {
        output.copy_from_slice(input);
        return;
    }
    let r_top = (kernel_h - 1) / 2;
    let r_bot = kernel_h / 2;

    for x in 0..w {
        let mut count = 0usize;

        // Initialize window covering [0, min(r_bot, h-1)].
        for y in 0..=r_bot.min(h - 1) {
            if input[y * w + x] > 0 {
                count += 1;
            }
        }

        for y in 0..h {
            output[y * w + x] = if count > 0 { 255 } else { 0 };

            let enter = y + r_bot + 1;
            if enter < h && input[enter * w + x] > 0 {
                count += 1;
            }

            if y >= r_top && input[(y - r_top) * w + x] > 0 {
                count -= 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_white_rgb(w: usize, h: usize) -> Vec<u8> {
        vec![255u8; w * h * 3]
    }

    fn make_black_rgb(w: usize, h: usize) -> Vec<u8> {
        vec![0u8; w * h * 3]
    }

    /// Helper to create a detector with no margins (for unit tests on small images).
    fn test_detector() -> MorphologicalDetector {
        MorphologicalDetector {
            side_margin: 0,
            top_margin: 0,
            bottom_margin: 0,
            filter_rotation: false,
            ..Default::default()
        }
    }

    #[test]
    fn blank_page_no_regions() {
        let img = make_white_rgb(100, 100);
        let det = test_detector();
        let regions = det.detect(&img, 100, 100, 0);
        assert!(regions.is_empty());
    }

    #[test]
    fn solid_black_page_no_regions() {
        // A uniform image (all same color) produces no foreground with adaptive
        // thresholding because there's no local contrast.
        let img = make_black_rgb(200, 200);
        let det = MorphologicalDetector {
            min_width: 1,
            min_height: 1,
            side_margin: 0,
            top_margin: 0,
            bottom_margin: 0,
            filter_rotation: false,
            ..Default::default()
        };
        let regions = det.detect(&img, 200, 200, 0);
        assert!(regions.is_empty());
    }

    #[test]
    fn horizontal_dilation_connects_nearby_pixels() {
        // Two pixels 5 apart in a row, kernel_w=20 should connect them.
        let mut input = vec![0u8; 100];
        input[42] = 255;
        input[47] = 255;
        let mut output = vec![0u8; 100];
        dilate_horizontal(&input, &mut output, 100, 1, 20);
        for x in 32..=56 {
            assert_eq!(output[x], 255, "pixel {x} should be dilated");
        }
        assert_eq!(output[31], 0, "pixel 31 should not be dilated");
        assert_eq!(output[57], 0, "pixel 57 should not be dilated");
    }

    #[test]
    fn vertical_dilation_small_kernel() {
        let mut input = vec![0u8; 25]; // 5x5
        input[12] = 255; // center (2,2)
        let mut output = vec![0u8; 25];
        dilate_vertical(&input, &mut output, 5, 5, 3);
        assert_eq!(output[7], 255); // (2,1)
        assert_eq!(output[12], 255); // (2,2)
        assert_eq!(output[17], 255); // (2,3)
        assert_eq!(output[2], 0);
        assert_eq!(output[22], 0);
    }

    #[test]
    fn text_line_detection() {
        // Simulate a single line of dark text on white background
        let w = 500;
        let h = 100;
        let mut img = vec![255u8; w * h * 3]; // white background
        // Draw a "text line" at y=40..60, x=50..400
        for y in 40..60 {
            for x in 50..400 {
                if x % 30 < 25 {
                    let idx = (y * w + x) * 3;
                    img[idx] = 0;
                    img[idx + 1] = 0;
                    img[idx + 2] = 0;
                }
            }
        }
        let det = MorphologicalDetector {
            min_width: 10,
            min_height: 10,
            side_margin: 0,
            top_margin: 0,
            bottom_margin: 0,
            filter_rotation: false,
            ..Default::default()
        };
        let regions = det.detect(&img, w as u32, h as u32, 0);
        assert!(!regions.is_empty());
        assert!(regions.len() <= 3, "dilation should merge nearby chars");
    }

    #[test]
    fn gaussian_blur_smooths() {
        // Single bright pixel in center of dark image.
        let w = 11;
        let h = 11;
        let mut img = vec![0u8; w * h];
        img[5 * w + 5] = 255;
        let mut out = vec![0u8; w * h];
        gaussian_blur(&img, &mut out, w, h, 5);
        // Center should be dimmed (spread out).
        assert!(out[5 * w + 5] < 255);
        // Neighbors should have some value.
        assert!(out[5 * w + 4] > 0);
        assert!(out[4 * w + 5] > 0);
    }

    #[test]
    fn adaptive_threshold_inverts_dark_text() {
        let w = 50;
        let h = 10;
        // Light background.
        let mut gray = vec![200u8; w * h];
        // Dark text pixels.
        for x in 10..40 {
            gray[5 * w + x] = 20;
        }
        let mut out = vec![0u8; w * h];
        adaptive_threshold_mean(&gray, &mut out, w, h, 15, 4.0);
        // Dark text should become white foreground.
        assert_eq!(out[5 * w + 20], 255);
        // Light background should be black.
        assert_eq!(out[0], 0);
    }

    #[test]
    fn margin_filtering_50_percent() {
        let det = MorphologicalDetector {
            side_margin: 100,
            top_margin: 100,
            bottom_margin: 100,
            ..Default::default()
        };
        let page_w = 1000.0;
        let page_h = 1000.0;

        // Region mostly inside left margin → filtered.
        let bbox_left = BBox::new(20.0, 200.0, 60.0, 30.0);
        assert!(!det.passes_margin_filter(&bbox_left, page_w, page_h));

        // Region straddling left margin but >50% past it → kept.
        let bbox_straddle = BBox::new(50.0, 200.0, 200.0, 30.0);
        assert!(det.passes_margin_filter(&bbox_straddle, page_w, page_h));

        // Region entirely in right margin → filtered.
        let bbox_right = BBox::new(920.0, 200.0, 60.0, 30.0);
        assert!(!det.passes_margin_filter(&bbox_right, page_w, page_h));

        // Region in top margin, <50% overflow → filtered.
        let bbox_top = BBox::new(200.0, 70.0, 100.0, 40.0);
        assert!(!det.passes_margin_filter(&bbox_top, page_w, page_h));

        // Interior region → kept.
        let bbox_ok = BBox::new(200.0, 200.0, 300.0, 30.0);
        assert!(det.passes_margin_filter(&bbox_ok, page_w, page_h));
    }

    #[test]
    fn row_column_clustering() {
        let det = test_detector();

        let mut regions = vec![
            // Row 0: two regions
            make_region(100.0, 100.0, 200.0, 30.0),
            make_region(400.0, 105.0, 150.0, 30.0),
            // Row 1 (close to row 0 → same cluster)
            make_region(100.0, 160.0, 200.0, 30.0),
            make_region(400.0, 162.0, 150.0, 30.0),
            // Row 2 (large gap → new cluster)
            make_region(100.0, 400.0, 200.0, 30.0),
        ];

        det.assign_rows_and_columns(&mut regions);

        // Row assignments
        assert_eq!(regions[0].row, Some(0));
        assert_eq!(regions[1].row, Some(0));
        assert_eq!(regions[2].row, Some(1));
        assert_eq!(regions[3].row, Some(1));
        assert_eq!(regions[4].row, Some(2));

        // Column assignments
        assert_eq!(regions[0].column, Some(0));
        assert_eq!(regions[1].column, Some(1));

        // Cluster assignments: rows 0-1 in cluster 0, row 2 in cluster 1
        assert_eq!(regions[0].cluster_id, Some(0));
        assert_eq!(regions[2].cluster_id, Some(0));
        assert_eq!(regions[4].cluster_id, Some(1));
    }

    fn make_region(x: f32, y: f32, w: f32, h: f32) -> TextRegion {
        TextRegion {
            bbox: BBox::new(x, y, w, h),
            polygon: Polygon {
                points: vec![(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
            },
            confidence: 1.0,
            page_index: 0,
            row: None,
            column: None,
            cluster_id: None,
        }
    }
}
