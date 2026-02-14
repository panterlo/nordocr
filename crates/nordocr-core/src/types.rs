use serde::{Deserialize, Serialize};

/// Axis-aligned bounding box in pixel coordinates.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl BBox {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    pub fn center(&self) -> (f32, f32) {
        (self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    pub fn right(&self) -> f32 {
        self.x + self.width
    }

    pub fn bottom(&self) -> f32 {
        self.y + self.height
    }

    /// Intersection-over-union with another bbox.
    pub fn iou(&self, other: &BBox) -> f32 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = self.right().min(other.right());
        let y2 = self.bottom().min(other.bottom());

        if x2 <= x1 || y2 <= y1 {
            return 0.0;
        }

        let intersection = (x2 - x1) * (y2 - y1);
        let union = self.area() + other.area() - intersection;

        if union <= 0.0 {
            0.0
        } else {
            intersection / union
        }
    }
}

/// Oriented bounding box represented as four corner points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Polygon {
    pub points: Vec<(f32, f32)>,
}

impl Polygon {
    pub fn to_bbox(&self) -> BBox {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        for &(x, y) in &self.points {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }

        BBox::new(min_x, min_y, max_x - min_x, max_y - min_y)
    }
}

/// A single recognized word within a text line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Word {
    pub text: String,
    pub confidence: f32,
    pub bbox: BBox,
}

/// A detected and recognized text line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextLine {
    pub text: String,
    pub confidence: f32,
    pub bbox: BBox,
    pub words: Option<Vec<Word>>,
}

/// OCR results for a single page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageResult {
    pub page_index: u32,
    pub text: String,
    pub lines: Vec<TextLine>,
    pub confidence: f32,
}

/// Timing information for pipeline stages.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimingInfo {
    pub decode_ms: f32,
    pub preprocess_ms: f32,
    pub detect_ms: f32,
    pub recognize_ms: f32,
    pub total_ms: f32,
}

/// Raw image data before GPU upload.
#[derive(Debug, Clone)]
pub struct RawImage {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}

/// Detected text region before recognition.
#[derive(Debug, Clone)]
pub struct TextRegion {
    pub bbox: BBox,
    pub polygon: Polygon,
    pub confidence: f32,
    pub page_index: u32,
}

/// Input file type discrimination.
#[derive(Debug, Clone)]
pub enum FileInput {
    Pdf(Vec<u8>),
    Image(Vec<u8>),
    MultiPageTiff(Vec<u8>),
}

impl FileInput {
    /// Detect file type from magic bytes.
    pub fn from_bytes(data: Vec<u8>) -> Self {
        if data.len() >= 4 {
            // PDF magic: %PDF
            if data.starts_with(b"%PDF") {
                return FileInput::Pdf(data);
            }
            // TIFF magic: II or MM
            if (data[0] == 0x49 && data[1] == 0x49 && data[2] == 0x2A && data[3] == 0x00)
                || (data[0] == 0x4D
                    && data[1] == 0x4D
                    && data[2] == 0x00
                    && data[3] == 0x2A)
            {
                // Could be multi-page TIFF — caller determines from page count
                return FileInput::MultiPageTiff(data);
            }
        }
        FileInput::Image(data)
    }
}
