use nordocr_core::{BBox, Polygon, TextRegion};

use crate::contour;

/// CPU-based morphological text region detector.
///
/// Algorithm (ported from Ormeo.Document `TextRegionOcrStep.cs`):
/// 1. RGB → grayscale
/// 2. Fixed threshold + invert (dark text → white foreground)
/// 3. Separable binary dilation with rect kernel (connects chars into lines)
/// 4. Connected component labeling → bounding boxes
/// 5. Filter by size and margin
/// 6. Sort top-to-bottom, left-to-right
pub struct MorphologicalDetector {
    /// Dilation kernel width (connects characters horizontally).
    pub kernel_w: u32,
    /// Dilation kernel height (keeps text lines separate).
    pub kernel_h: u32,
    /// Number of dilation iterations.
    pub iterations: u32,
    /// Binary threshold: pixels with grayscale < threshold become foreground.
    pub threshold: u8,
    /// Minimum bounding box width to keep.
    pub min_width: u32,
    /// Minimum bounding box height to keep.
    pub min_height: u32,
    /// Margin in pixels to skip at left/right page edges.
    pub margin_x: u32,
    /// Margin in pixels to skip at top/bottom page edges.
    pub margin_y: u32,
    /// Padding added around each detected region (pixels).
    pub region_padding: u32,
}

impl Default for MorphologicalDetector {
    /// Defaults from the Ormeo.Document C# pipeline (`CreateDefaultOcrSettings`).
    fn default() -> Self {
        Self {
            kernel_w: 20,
            kernel_h: 3,
            iterations: 2,
            threshold: 128,
            min_width: 20,
            min_height: 24,
            margin_x: 0,
            margin_y: 0,
            region_padding: 5,
        }
    }
}

impl MorphologicalDetector {
    /// Detect text regions in an RGB u8 HWC image (CPU memory).
    ///
    /// Returns text regions sorted top-to-bottom, left-to-right.
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

        // Step 2: Threshold + invert (dark text becomes white foreground).
        let mut binary = vec![0u8; n];
        let thresh = self.threshold;
        for i in 0..n {
            binary[i] = if gray[i] < thresh { 255 } else { 0 };
        }
        drop(gray);

        // Step 3: Separable binary dilation.
        let mut current = binary;
        for _ in 0..self.iterations {
            let mut h_dilated = vec![0u8; n];
            dilate_horizontal(&current, &mut h_dilated, w, h, self.kernel_w as usize);
            let mut v_dilated = vec![0u8; n];
            dilate_vertical(&h_dilated, &mut v_dilated, w, h, self.kernel_h as usize);
            current = v_dilated;
        }

        // Step 4: CCL → bounding boxes.
        let bboxes = contour::extract_bboxes(&current, width, height);
        drop(current);

        tracing::debug!(
            page = page_index,
            raw_regions = bboxes.len(),
            "morphological detection: raw components"
        );

        // Step 5: Filter by size and margin.
        let margin_x = self.margin_x as f32;
        let margin_y = self.margin_y as f32;
        let min_w = self.min_width as f32;
        let min_h = self.min_height as f32;
        let page_w = width as f32;
        let page_h = height as f32;

        let mut regions: Vec<TextRegion> = bboxes
            .into_iter()
            .filter(|b| {
                b.width >= min_w
                    && b.height >= min_h
                    && b.x >= margin_x
                    && b.y >= margin_y
                    && (b.x + b.width) <= (page_w - margin_x)
                    && (b.y + b.height) <= (page_h - margin_y)
            })
            .map(|b| {
                let pad = self.region_padding as f32;
                let x = (b.x - pad).max(0.0);
                let y = (b.y - pad).max(0.0);
                let w = (b.width + 2.0 * pad).min(page_w - x);
                let h = (b.height + 2.0 * pad).min(page_h - y);
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
                }
            })
            .collect();

        // Step 6: Sort top-to-bottom, left-to-right.
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

        regions
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

    #[test]
    fn blank_page_no_regions() {
        let img = make_white_rgb(100, 100);
        let det = MorphologicalDetector::default();
        let regions = det.detect(&img, 100, 100, 0);
        assert!(regions.is_empty());
    }

    #[test]
    fn solid_black_page_one_region() {
        let img = make_black_rgb(200, 200);
        let det = MorphologicalDetector {
            min_width: 1,
            min_height: 1,
            ..Default::default()
        };
        let regions = det.detect(&img, 200, 200, 0);
        assert!(!regions.is_empty());
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
        assert_eq!(output[7], 255);  // (2,1)
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
            ..Default::default()
        };
        let regions = det.detect(&img, w as u32, h as u32, 0);
        assert!(!regions.is_empty());
        assert!(regions.len() <= 3, "dilation should merge nearby chars");
    }
}
