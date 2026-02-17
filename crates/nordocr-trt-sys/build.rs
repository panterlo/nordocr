use std::env;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // --- Feature: generate bindgen bindings from full TensorRT headers ---
    // This is the original approach, kept for reference.
    if env::var("CARGO_FEATURE_GENERATED").is_ok() {
        let trt_lib_path = env::var("TENSORRT_LIB_DIR")
            .unwrap_or_else(|_| "/usr/lib/x86_64-linux-gnu".to_string());
        let trt_include_path =
            env::var("TENSORRT_INCLUDE_DIR").unwrap_or_else(|_| "/usr/include".to_string());

        println!("cargo:rustc-link-lib=nvinfer");
        println!("cargo:rustc-link-lib=nvinfer_plugin");
        println!("cargo:rustc-link-search=native={trt_lib_path}");
        println!("cargo:rerun-if-env-changed=TENSORRT_LIB_DIR");
        println!("cargo:rerun-if-env-changed=TENSORRT_INCLUDE_DIR");

        let header = format!("{trt_include_path}/NvInfer.h");
        println!("cargo:rerun-if-changed={header}");

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

        bindings
            .write_to_file(out_dir.join("trt_bindings.rs"))
            .expect("Failed to write TensorRT bindings");

        return;
    }

    // --- Default: Compile the C++ shim against bundled TRT headers ---
    // This is the primary build path for development and production.

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let include_dir = manifest_dir.join("include");
    let shim_cpp = manifest_dir.join("trt_shim.cpp");

    // Check if the C++ shim and headers exist.
    if !shim_cpp.exists() || !include_dir.join("NvInferRuntime.h").exists() {
        // No shim available — build in stub mode (no TRT operations at runtime).
        println!("cargo:warning=TRT shim not found — building in stub mode");
        return;
    }

    println!("cargo:rerun-if-changed=trt_shim.cpp");
    println!("cargo:rerun-if-changed=include/NvInferRuntime.h");
    println!("cargo:rerun-if-changed=include/NvInferRuntimeBase.h");
    println!("cargo:rerun-if-changed=include/NvInferVersion.h");
    println!("cargo:rerun-if-env-changed=TENSORRT_LIB_DIR");

    // Find TensorRT shared library location.
    let trt_lib_dir = find_trt_lib_dir();

    // Find CUDA include directory (needed for cuda_runtime_api.h).
    let cuda_include = find_cuda_include_dir();

    // Compile the C++ shim.
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file(&shim_cpp)
        .include(&include_dir)
        .warnings(false)
        .opt_level(2);

    if let Some(cuda_inc) = &cuda_include {
        build.include(cuda_inc);
    }

    // Platform-specific C++ standard flag.
    if cfg!(target_os = "windows") {
        build.flag("/std:c++17").flag("/EHsc");
    } else {
        build.flag("-std=c++17");
    }

    build.compile("trt_shim");

    // Link against TensorRT runtime library.
    if let Some(lib_dir) = &trt_lib_dir {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }

    if cfg!(target_os = "windows") {
        // On Windows, link to nvinfer_10.lib (import library).
        if let Some(lib_dir) = &trt_lib_dir {
            let lib_file = lib_dir.join("nvinfer_10.lib");
            if lib_file.exists() {
                println!("cargo:rustc-link-lib=nvinfer_10");
            } else {
                generate_import_lib(lib_dir, &out_dir);
                println!("cargo:rustc-link-search=native={}", out_dir.display());
                println!("cargo:rustc-link-lib=nvinfer_10");
            }
        }
    } else {
        // On Linux, link to libnvinfer.so.10
        println!("cargo:rustc-link-lib=nvinfer");
    }

    // On non-MSVC platforms, need libstdc++ for the C++ shim.
    // MSVC links the C++ runtime automatically.
    if !cfg!(target_env = "msvc") {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

/// Find the CUDA include directory containing cuda_runtime_api.h.
fn find_cuda_include_dir() -> Option<PathBuf> {
    // 1. CUDA_ROOT or CUDA_PATH environment variable.
    for var in ["CUDA_ROOT", "CUDA_PATH"] {
        if let Ok(cuda_path) = env::var(var) {
            let p = PathBuf::from(&cuda_path).join("include");
            if p.join("cuda_runtime_api.h").exists() {
                return Some(p);
            }
        }
    }

    // 2. On Linux, check standard paths.
    if cfg!(target_os = "linux") {
        for dir in ["/usr/local/cuda-13.0/include", "/usr/local/cuda/include", "/usr/include"] {
            let p = PathBuf::from(dir);
            if p.join("cuda_runtime_api.h").exists() {
                return Some(p);
            }
        }
    }

    None
}

/// Find the TensorRT library directory.
fn find_trt_lib_dir() -> Option<PathBuf> {
    // 1. Check TENSORRT_LIB_DIR environment variable.
    if let Ok(dir) = env::var("TENSORRT_LIB_DIR") {
        let p = PathBuf::from(dir);
        if p.exists() {
            return Some(p);
        }
    }

    // 2. On Linux, check standard paths.
    if cfg!(target_os = "linux") {
        for dir in [
            "/usr/lib/x86_64-linux-gnu",
            "/usr/local/lib",
            "/usr/lib",
        ] {
            let p = PathBuf::from(dir);
            if p.join("libnvinfer.so.10").exists() || p.join("libnvinfer.so").exists() {
                return Some(p);
            }
        }
    }

    None
}

/// Generate a .lib import library from a .dll on Windows.
fn generate_import_lib(dll_dir: &PathBuf, out_dir: &PathBuf) {
    let _dll_path = dll_dir.join("nvinfer_10.dll");
    let def_path = out_dir.join("nvinfer_10.def");
    let lib_path = out_dir.join("nvinfer_10.lib");

    if lib_path.exists() {
        return;
    }

    let def_content = "\
LIBRARY nvinfer_10.dll
EXPORTS
    createInferRuntime_INTERNAL
    createInferBuilder_INTERNAL
    getInferLibVersion
    getInferLibMajorVersion
    getInferLibMinorVersion
    getInferLibPatchVersion
    getInferLibBuildVersion
    getPluginRegistry
    getLogger
";
    std::fs::write(&def_path, def_content).expect("Failed to write .def file");

    let status = std::process::Command::new("lib")
        .arg(format!("/DEF:{}", def_path.display()))
        .arg(format!("/OUT:{}", lib_path.display()))
        .arg("/MACHINE:X64")
        .arg("/NOLOGO")
        .status();

    match status {
        Ok(s) if s.success() => {
            println!(
                "cargo:warning=Generated import library: {}",
                lib_path.display()
            );
        }
        _ => {
            println!("cargo:warning=Failed to generate import library — TRT shim will not link. Set TENSORRT_LIB_DIR.");
        }
    }
}
