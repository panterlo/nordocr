//! Tesseract LSTM recognition backend via runtime DLL loading.
//!
//! Loads `tesseract55.dll` (or equivalent) at runtime using `libloading`,
//! resolves the C API symbols, and provides a thread-safe recognizer that
//! runs Tesseract OCR on CPU-cropped text regions in parallel via rayon.

use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;
use std::sync::Arc;

use nordocr_core::{BBox, OcrError, RawImage, Result, TextLine, TextRegion};
use rayon::prelude::*;
use tracing;

/// Page segmentation mode: treat the image as a single text line.
const PSM_SINGLE_LINE: c_int = 7;

/// Loaded Tesseract C API function pointers.
struct TessApi {
    _lib: libloading::Library,
    create: unsafe extern "C" fn() -> *mut c_void,
    init3: unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char) -> c_int,
    set_image: unsafe extern "C" fn(*mut c_void, *const u8, c_int, c_int, c_int, c_int),
    set_page_seg_mode: unsafe extern "C" fn(*mut c_void, c_int),
    recognize: unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int,
    get_utf8_text: unsafe extern "C" fn(*mut c_void) -> *mut c_char,
    mean_text_conf: unsafe extern "C" fn(*mut c_void) -> c_int,
    clear: unsafe extern "C" fn(*mut c_void),
    end: unsafe extern "C" fn(*mut c_void),
    delete: unsafe extern "C" fn(*mut c_void),
    delete_text: unsafe extern "C" fn(*mut c_char),
}

// Safety: The TessApi struct holds function pointers and an owned Library.
// The function pointers are valid for the lifetime of _lib. We only call
// them through per-thread TessBaseAPI instances (create/init per thread),
// so there is no shared mutable state across threads.
unsafe impl Send for TessApi {}
unsafe impl Sync for TessApi {}

impl TessApi {
    /// Load the Tesseract DLL and resolve all required C API symbols.
    fn load(dll_path: &Path) -> Result<Self> {
        // Add the DLL's directory to the search path so that dependent DLLs
        // (e.g. leptonica-1.87.0.dll) in the same directory are found.
        if let Some(dir) = dll_path.parent() {
            #[cfg(windows)]
            {
                use std::os::windows::ffi::OsStrExt;
                let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(Some(0)).collect();
                // Safety: SetDllDirectoryW is a standard Windows API call.
                unsafe {
                    extern "system" {
                        fn SetDllDirectoryW(path: *const u16) -> i32;
                    }
                    SetDllDirectoryW(wide.as_ptr());
                }
            }
            let _ = dir; // suppress unused on non-Windows
        }

        // Safety: Loading a dynamic library is inherently unsafe. We trust that
        // the user-provided DLL path points to a valid Tesseract library.
        let lib = unsafe { libloading::Library::new(dll_path) }.map_err(|e| {
            OcrError::ModelLoad(format!(
                "failed to load Tesseract DLL '{}': {}",
                dll_path.display(),
                e
            ))
        })?;

        // Resolve symbols. Tesseract C API exports these as plain C names.
        unsafe {
            let create = *lib
                .get::<unsafe extern "C" fn() -> *mut c_void>(b"TessBaseAPICreate\0")
                .map_err(|e| OcrError::ModelLoad(format!("TessBaseAPICreate: {e}")))?;
            let init3 = *lib
                .get::<unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char) -> c_int>(
                    b"TessBaseAPIInit3\0",
                )
                .map_err(|e| OcrError::ModelLoad(format!("TessBaseAPIInit3: {e}")))?;
            let set_image = *lib
                .get::<unsafe extern "C" fn(*mut c_void, *const u8, c_int, c_int, c_int, c_int)>(
                    b"TessBaseAPISetImage\0",
                )
                .map_err(|e| OcrError::ModelLoad(format!("TessBaseAPISetImage: {e}")))?;
            let set_page_seg_mode = *lib
                .get::<unsafe extern "C" fn(*mut c_void, c_int)>(b"TessBaseAPISetPageSegMode\0")
                .map_err(|e| OcrError::ModelLoad(format!("TessBaseAPISetPageSegMode: {e}")))?;
            let recognize = *lib
                .get::<unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int>(
                    b"TessBaseAPIRecognize\0",
                )
                .map_err(|e| OcrError::ModelLoad(format!("TessBaseAPIRecognize: {e}")))?;
            let get_utf8_text = *lib
                .get::<unsafe extern "C" fn(*mut c_void) -> *mut c_char>(
                    b"TessBaseAPIGetUTF8Text\0",
                )
                .map_err(|e| OcrError::ModelLoad(format!("TessBaseAPIGetUTF8Text: {e}")))?;
            let mean_text_conf = *lib
                .get::<unsafe extern "C" fn(*mut c_void) -> c_int>(
                    b"TessBaseAPIMeanTextConf\0",
                )
                .map_err(|e| OcrError::ModelLoad(format!("TessBaseAPIMeanTextConf: {e}")))?;
            let clear = *lib
                .get::<unsafe extern "C" fn(*mut c_void)>(b"TessBaseAPIClear\0")
                .map_err(|e| OcrError::ModelLoad(format!("TessBaseAPIClear: {e}")))?;
            let end = *lib
                .get::<unsafe extern "C" fn(*mut c_void)>(b"TessBaseAPIEnd\0")
                .map_err(|e| OcrError::ModelLoad(format!("TessBaseAPIEnd: {e}")))?;
            let delete = *lib
                .get::<unsafe extern "C" fn(*mut c_void)>(b"TessBaseAPIDelete\0")
                .map_err(|e| OcrError::ModelLoad(format!("TessBaseAPIDelete: {e}")))?;
            let delete_text = *lib
                .get::<unsafe extern "C" fn(*mut c_char)>(b"TessDeleteText\0")
                .map_err(|e| OcrError::ModelLoad(format!("TessDeleteText: {e}")))?;

            Ok(Self {
                _lib: lib,
                create,
                init3,
                set_image,
                set_page_seg_mode,
                recognize,
                get_utf8_text,
                mean_text_conf,
                clear,
                end,
                delete,
                delete_text,
            })
        }
    }
}

/// A per-thread Tesseract API instance. Not Send/Sync — lives in thread_local.
struct TessInstance {
    handle: *mut c_void,
    api: Arc<TessApi>,
}

impl TessInstance {
    fn new(api: Arc<TessApi>, tessdata: &CStr, language: &CStr) -> Result<Self> {
        let handle = unsafe { (api.create)() };
        if handle.is_null() {
            return Err(OcrError::Recognition(
                "TessBaseAPICreate returned null".into(),
            ));
        }
        let rc = unsafe { (api.init3)(handle, tessdata.as_ptr(), language.as_ptr()) };
        if rc != 0 {
            unsafe { (api.delete)(handle) };
            return Err(OcrError::Recognition(format!(
                "TessBaseAPIInit3 failed (rc={}), tessdata={:?} lang={:?}",
                rc,
                tessdata,
                language,
            )));
        }
        // Set single-line mode once; persists across Clear() calls.
        unsafe { (api.set_page_seg_mode)(handle, PSM_SINGLE_LINE) };
        Ok(Self { handle, api })
    }

    /// Recognize a single cropped region image.
    fn recognize_region(&self, pixels: &[u8], width: u32, height: u32) -> Result<(String, f32)> {
        unsafe {
            (self.api.set_image)(
                self.handle,
                pixels.as_ptr(),
                width as c_int,
                height as c_int,
                3, // bytes per pixel (RGB)
                (width * 3) as c_int,
            );
            let rc = (self.api.recognize)(self.handle, std::ptr::null_mut());
            if rc != 0 {
                (self.api.clear)(self.handle);
                return Err(OcrError::Recognition(format!(
                    "TessBaseAPIRecognize failed (rc={})",
                    rc
                )));
            }

            let text_ptr = (self.api.get_utf8_text)(self.handle);
            let conf = (self.api.mean_text_conf)(self.handle);

            let text = if text_ptr.is_null() {
                String::new()
            } else {
                let s = CStr::from_ptr(text_ptr).to_string_lossy().into_owned();
                (self.api.delete_text)(text_ptr);
                s.trim().to_string()
            };

            (self.api.clear)(self.handle);

            Ok((text, conf.max(0) as f32 / 100.0))
        }
    }
}

impl Drop for TessInstance {
    fn drop(&mut self) {
        unsafe {
            (self.api.end)(self.handle);
            (self.api.delete)(self.handle);
        }
    }
}

/// Thread-safe Tesseract recognizer. Uses per-thread API instances via
/// `thread_local!` so that rayon workers each get their own Tesseract context.
pub struct TesseractRecognizer {
    api: Arc<TessApi>,
    tessdata_path: CString,
    language: CString,
}

impl TesseractRecognizer {
    /// Create a new Tesseract recognizer.
    ///
    /// - `dll_path`: Path to `tesseract55.dll` (or equivalent).
    /// - `tessdata_path`: Directory containing `.traineddata` files.
    /// - `language`: Model name (e.g. "swe_ormeo_v3").
    pub fn new(dll_path: &Path, tessdata_path: &str, language: &str) -> Result<Self> {
        let api = Arc::new(TessApi::load(dll_path)?);

        let tessdata_cstr = CString::new(tessdata_path).map_err(|e| {
            OcrError::InvalidInput(format!("invalid tessdata path: {e}"))
        })?;
        let language_cstr = CString::new(language).map_err(|e| {
            OcrError::InvalidInput(format!("invalid language name: {e}"))
        })?;

        // Validate by creating one instance eagerly.
        let test = TessInstance::new(Arc::clone(&api), &tessdata_cstr, &language_cstr)?;
        drop(test);

        tracing::info!(
            dll = %dll_path.display(),
            tessdata = tessdata_path,
            lang = language,
            "Tesseract recognizer initialized"
        );

        Ok(Self {
            api,
            tessdata_path: tessdata_cstr,
            language: language_cstr,
        })
    }

    /// Recognize all detected text regions using Tesseract.
    ///
    /// Crops each region from its page image and runs Tesseract OCR in parallel.
    pub fn recognize_all(
        &self,
        regions: &[TextRegion],
        page_images: &[RawImage],
    ) -> Result<Vec<TextLine>> {
        if regions.is_empty() {
            return Ok(Vec::new());
        }

        // Clone Arc references for thread_local initialization.
        let api = Arc::clone(&self.api);
        let tessdata = self.tessdata_path.clone();
        let language = self.language.clone();

        thread_local! {
            static TESS: RefCell<Option<TessInstance>> = const { RefCell::new(None) };
        }

        let results: Vec<Result<TextLine>> = regions
            .par_iter()
            .map(|region| {
                // Crop the region from the page image.
                let page = page_images
                    .get(region.page_index as usize)
                    .ok_or_else(|| {
                        OcrError::Recognition(format!(
                            "page index {} out of range (have {})",
                            region.page_index,
                            page_images.len()
                        ))
                    })?;

                let crop = crop_region(page, &region.bbox);
                let (crop_w, crop_h, crop_pixels) = crop;

                if crop_w == 0 || crop_h == 0 {
                    return Ok(TextLine {
                        text: String::new(),
                        confidence: 0.0,
                        bbox: region.bbox.clone(),
                        words: None,
                        char_confidences: Vec::new(),
                    });
                }

                // Get or initialize the per-thread Tesseract instance.
                TESS.with(|cell| {
                    let mut slot = cell.borrow_mut();
                    if slot.is_none() {
                        let inst = TessInstance::new(
                            Arc::clone(&api),
                            &tessdata,
                            &language,
                        )?;
                        *slot = Some(inst);
                    }
                    let inst = slot.as_ref().unwrap();
                    let (text, confidence) =
                        inst.recognize_region(&crop_pixels, crop_w, crop_h)?;

                    Ok(TextLine {
                        text,
                        confidence,
                        bbox: region.bbox.clone(),
                        words: None,
                        char_confidences: Vec::new(),
                    })
                })
            })
            .collect();

        // Collect results, propagating first error.
        results.into_iter().collect()
    }
}

/// Crop a region from a page image. Returns (width, height, RGB pixels).
fn crop_region(page: &RawImage, bbox: &BBox) -> (u32, u32, Vec<u8>) {
    let x0 = (bbox.x as u32).min(page.width);
    let y0 = (bbox.y as u32).min(page.height);
    let x1 = ((bbox.x + bbox.width) as u32).min(page.width);
    let y1 = ((bbox.y + bbox.height) as u32).min(page.height);

    let w = x1.saturating_sub(x0);
    let h = y1.saturating_sub(y0);

    if w == 0 || h == 0 {
        return (0, 0, Vec::new());
    }

    let channels = page.channels as usize;
    let src_stride = page.width as usize * channels;
    let dst_stride = w as usize * channels;
    let mut pixels = vec![0u8; h as usize * dst_stride];

    for row in 0..h as usize {
        let src_start = (y0 as usize + row) * src_stride + x0 as usize * channels;
        let src_end = src_start + dst_stride;
        let dst_start = row * dst_stride;
        pixels[dst_start..dst_start + dst_stride]
            .copy_from_slice(&page.data[src_start..src_end]);
    }

    (w, h, pixels)
}
