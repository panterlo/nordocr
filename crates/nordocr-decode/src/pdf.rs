use nordocr_core::{OcrError, RawImage, Result};

/// Renders PDF pages to images for OCR processing.
///
/// Uses `pdfium-render` for high-quality PDF rendering. Each page is
/// rendered at a configurable DPI, then passed to the GPU pipeline.
pub struct PdfDecoder {
    render_dpi: f32,
}

impl PdfDecoder {
    pub fn new(render_dpi: f32) -> Self {
        Self { render_dpi }
    }

    /// Render all pages (or a subset) of a PDF to images.
    pub fn render_pages(
        &self,
        pdf_data: &[u8],
        page_indices: Option<&[u32]>,
    ) -> Result<Vec<RawImage>> {
        // In production with pdfium-render:
        //
        //   let pdfium = Pdfium::default();
        //   let doc = pdfium.load_pdf_from_byte_slice(pdf_data, None)
        //       .map_err(|e| OcrError::PdfRender(e.to_string()))?;
        //
        //   let pages_to_render: Vec<u32> = match page_indices {
        //       Some(indices) => indices.to_vec(),
        //       None => (0..doc.pages().len() as u32).collect(),
        //   };
        //
        //   let mut images = Vec::new();
        //   for &page_idx in &pages_to_render {
        //       let page = doc.pages().get(page_idx as u16)
        //           .map_err(|e| OcrError::PdfRender(e.to_string()))?;
        //
        //       let render_config = PdfRenderConfig::new()
        //           .set_target_width(
        //               (page.width().inches() * self.render_dpi) as i32
        //           )
        //           .set_target_height(
        //               (page.height().inches() * self.render_dpi) as i32
        //           );
        //
        //       let bitmap = page.render_with_config(&render_config)
        //           .map_err(|e| OcrError::PdfRender(e.to_string()))?;
        //
        //       let img = bitmap.as_image();
        //       let gray = img.to_luma8();
        //
        //       images.push(RawImage {
        //           data: gray.clone().into_raw(),
        //           width: gray.width(),
        //           height: gray.height(),
        //           channels: 1,
        //       });
        //   }
        //
        //   Ok(images)

        let _ = (pdf_data, page_indices);

        tracing::info!(
            dpi = self.render_dpi,
            "PDF rendering requires pdfium library at runtime"
        );

        Err(OcrError::PdfRender(
            "pdfium library not loaded — PDF rendering not available in stub mode".into(),
        ))
    }

    /// Get the number of pages in a PDF without rendering.
    pub fn page_count(&self, pdf_data: &[u8]) -> Result<u32> {
        // In production:
        //   let doc = pdfium.load_pdf_from_byte_slice(pdf_data, None)?;
        //   Ok(doc.pages().len() as u32)

        let _ = pdf_data;
        Err(OcrError::PdfRender("pdfium not loaded".into()))
    }
}
