use nordocr_core::BBox;

/// Extract bounding boxes of connected components from a binary u8 image.
///
/// White pixels (> 0) are foreground, black (0) is background.
/// Uses 4-connected two-pass CCL with union-find (path halving).
pub fn extract_bboxes(binary: &[u8], width: u32, height: u32) -> Vec<BBox> {
    let w = width as usize;
    let h = height as usize;

    if w == 0 || h == 0 {
        return Vec::new();
    }

    let mut labels = vec![0u32; w * h];
    // parent[0] = 0 (background, unused). Grows as labels are assigned.
    let mut parent: Vec<u32> = vec![0];
    let mut next_label: u32 = 1;

    // --- Pass 1: assign provisional labels ---
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if binary[idx] == 0 {
                continue;
            }

            let left = if x > 0 { labels[idx - 1] } else { 0 };
            let top = if y > 0 { labels[idx - w] } else { 0 };

            match (left > 0, top > 0) {
                (false, false) => {
                    labels[idx] = next_label;
                    parent.push(next_label); // self-referencing root
                    next_label += 1;
                }
                (true, false) => {
                    labels[idx] = left;
                }
                (false, true) => {
                    labels[idx] = top;
                }
                (true, true) => {
                    let rl = find(&mut parent, left);
                    let rt = find(&mut parent, top);
                    let min_l = rl.min(rt);
                    labels[idx] = min_l;
                    if rl != rt {
                        let max_l = rl.max(rt);
                        parent[max_l as usize] = min_l;
                    }
                }
            }
        }
    }

    // Flatten all parent pointers to roots.
    for i in 1..parent.len() {
        parent[i] = find(&mut parent, i as u32);
    }

    // --- Pass 2: compute bounding boxes ---
    let mut bbox_map: Vec<Option<(u32, u32, u32, u32)>> = vec![None; parent.len()];

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let label = labels[idx];
            if label == 0 {
                continue;
            }
            let root = parent[label as usize] as usize;
            let xu = x as u32;
            let yu = y as u32;
            match bbox_map[root] {
                Some(ref mut e) => {
                    e.0 = e.0.min(xu);
                    e.1 = e.1.min(yu);
                    e.2 = e.2.max(xu);
                    e.3 = e.3.max(yu);
                }
                None => {
                    bbox_map[root] = Some((xu, yu, xu, yu));
                }
            }
        }
    }

    bbox_map
        .into_iter()
        .flatten()
        .map(|(x0, y0, x1, y1)| {
            BBox::new(x0 as f32, y0 as f32, (x1 - x0 + 1) as f32, (y1 - y0 + 1) as f32)
        })
        .collect()
}

/// Union-find with path halving.
fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_image() {
        let img = vec![0u8; 100];
        let bbs = extract_bboxes(&img, 10, 10);
        assert!(bbs.is_empty());
    }

    #[test]
    fn single_pixel() {
        let mut img = vec![0u8; 25];
        img[12] = 255; // center of 5x5
        let bbs = extract_bboxes(&img, 5, 5);
        assert_eq!(bbs.len(), 1);
        assert_eq!(bbs[0].x, 2.0);
        assert_eq!(bbs[0].y, 2.0);
        assert_eq!(bbs[0].width, 1.0);
        assert_eq!(bbs[0].height, 1.0);
    }

    #[test]
    fn two_separate_blobs() {
        // 10x5 image with two blobs
        let mut img = vec![0u8; 50];
        // Blob 1: top-left 2x2
        img[0] = 255;
        img[1] = 255;
        img[10] = 255;
        img[11] = 255;
        // Blob 2: bottom-right 3x1
        img[37] = 255;
        img[38] = 255;
        img[39] = 255;
        let bbs = extract_bboxes(&img, 10, 5);
        assert_eq!(bbs.len(), 2);
    }

    #[test]
    fn l_shaped_blob_merges() {
        // 5x5, L-shape that requires union
        let mut img = vec![0u8; 25];
        // Horizontal bar: row 0, cols 0-3
        for x in 0..4 {
            img[x] = 255;
        }
        // Vertical bar: col 0, rows 0-3
        for y in 0..4 {
            img[y * 5] = 255;
        }
        let bbs = extract_bboxes(&img, 5, 5);
        assert_eq!(bbs.len(), 1);
        assert_eq!(bbs[0].x, 0.0);
        assert_eq!(bbs[0].y, 0.0);
        assert_eq!(bbs[0].width, 4.0);
        assert_eq!(bbs[0].height, 4.0);
    }
}
