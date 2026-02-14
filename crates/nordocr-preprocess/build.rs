use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// GPU targets:
///   sm_89  — A6000 Ada (development)
///   sm_120 — RTX 6000 PRO Blackwell (production)
///
/// By default we compile fat binaries with both architectures so the
/// same binary runs on either GPU. The CUDA driver picks the best match
/// at load time, or JIT-compiles from the embedded PTX if needed.
///
/// Override via: NORDOCR_CUDA_ARCHS="sm_89,sm_120"
/// For dev-only builds: NORDOCR_CUDA_ARCHS="sm_89"
const DEFAULT_ARCHS: &str = "sm_89,sm_120";

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernels_dir = Path::new("kernels");

    let kernels = ["binarize", "deskew", "denoise", "morphology"];

    let archs_str =
        env::var("NORDOCR_CUDA_ARCHS").unwrap_or_else(|_| DEFAULT_ARCHS.to_string());
    let archs: Vec<&str> = archs_str.split(',').map(|s| s.trim()).collect();

    println!("cargo:rerun-if-env-changed=NORDOCR_CUDA_ARCHS");

    for kernel in &kernels {
        let cu_file = kernels_dir.join(format!("{kernel}.cu"));
        println!("cargo:rerun-if-changed={}", cu_file.display());

        // Compile a separate .ptx per architecture, named {kernel}_{arch}.ptx
        // e.g. binarize_sm_89.ptx, binarize_sm_120.ptx
        for arch in &archs {
            let safe_arch = arch.replace("sm_", "sm");
            let ptx_file = out_dir.join(format!("{kernel}_{safe_arch}.ptx"));

            // Extract compute capability number for --gencode.
            // sm_89 → compute_89, sm_120 → compute_120
            let compute = arch.replace("sm_", "compute_");

            let status = Command::new("nvcc")
                .args([
                    "--ptx",
                    &format!("--generate-code=arch={compute},code={arch}"),
                    "--use_fast_math",
                    "-O3",
                    "-o",
                    ptx_file.to_str().unwrap(),
                    cu_file.to_str().unwrap(),
                ])
                .status();

            match status {
                Ok(s) if s.success() => {
                    println!(
                        "cargo:warning=compiled {kernel}.cu → {kernel}_{safe_arch}.ptx (arch={arch})"
                    );
                }
                Ok(s) => {
                    eprintln!("nvcc failed for {kernel}.cu ({arch}) with status {s}, creating stub");
                    std::fs::write(
                        &ptx_file,
                        format!("// STUB: {kernel}.cu not compiled for {arch}\n"),
                    )
                    .unwrap();
                }
                Err(e) => {
                    eprintln!("nvcc not found ({e}), creating stub PTX for {kernel}.cu ({arch})");
                    std::fs::write(
                        &ptx_file,
                        format!("// STUB: {kernel}.cu not compiled (nvcc not found)\n"),
                    )
                    .unwrap();
                }
            }
        }

        // Also compile a fat cubin with all architectures embedded.
        // This is what gets loaded at runtime — CUDA driver picks the right code.
        let fatbin_file = out_dir.join(format!("{kernel}.fatbin"));
        let mut nvcc_args: Vec<String> = vec![
            "--fatbin".to_string(),
            "--use_fast_math".to_string(),
            "-O3".to_string(),
        ];
        for arch in &archs {
            let compute = arch.replace("sm_", "compute_");
            nvcc_args.push(format!("--generate-code=arch={compute},code={arch}"));
            // Also embed PTX for forward compat (JIT on future GPUs).
            nvcc_args.push(format!("--generate-code=arch={compute},code={compute}"));
        }
        nvcc_args.push("-o".to_string());
        nvcc_args.push(fatbin_file.to_str().unwrap().to_string());
        nvcc_args.push(cu_file.to_str().unwrap().to_string());

        let status = Command::new("nvcc")
            .args(&nvcc_args)
            .status();

        match status {
            Ok(s) if s.success() => {
                println!(
                    "cargo:warning=compiled {kernel}.cu → {kernel}.fatbin (archs={archs_str})"
                );
            }
            Ok(s) => {
                eprintln!("nvcc fatbin failed for {kernel}.cu with status {s}");
            }
            Err(_) => {
                // Already warned above for PTX stubs.
            }
        }
    }
}
