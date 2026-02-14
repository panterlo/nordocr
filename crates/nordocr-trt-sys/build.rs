use std::env;
use std::path::PathBuf;

fn main() {
    // Only generate bindings when the "generated" feature is active.
    // Without TensorRT installed, the crate uses stub types from src/lib.rs.
    if env::var("CARGO_FEATURE_GENERATED").is_err() {
        return;
    }

    // Link TensorRT libraries.
    println!("cargo:rustc-link-lib=nvinfer");
    println!("cargo:rustc-link-lib=nvinfer_plugin");

    // Search paths — configurable via environment variables.
    let trt_lib_path =
        env::var("TENSORRT_LIB_DIR").unwrap_or_else(|_| "/usr/lib/x86_64-linux-gnu".to_string());
    let trt_include_path =
        env::var("TENSORRT_INCLUDE_DIR").unwrap_or_else(|_| "/usr/include".to_string());

    println!("cargo:rustc-link-search=native={trt_lib_path}");
    println!("cargo:rerun-if-env-changed=TENSORRT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=TENSORRT_INCLUDE_DIR");

    let header = format!("{trt_include_path}/NvInfer.h");
    println!("cargo:rerun-if-changed={header}");

    // Generate bindings to TensorRT C API.
    let bindings = bindgen::Builder::default()
        .header(&header)
        .clang_arg(format!("-I{trt_include_path}"))
        .allowlist_function("nv.*")
        .allowlist_type("nv.*")
        .allowlist_var("nv.*")
        .layout_tests(false)
        .generate_comments(true)
        .generate()
        .expect("Failed to generate TensorRT bindings. Is TensorRT installed?");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("trt_bindings.rs"))
        .expect("Failed to write TensorRT bindings");
}
