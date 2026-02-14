use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernels_dir = Path::new("kernels");

    let kernels = ["binarize", "deskew", "denoise", "morphology"];

    // Target sm_100 (Blackwell) with fast math for maximum throughput.
    let nvcc_arch = env::var("NORDOCR_CUDA_ARCH").unwrap_or_else(|_| "sm_100".to_string());

    for kernel in &kernels {
        let cu_file = kernels_dir.join(format!("{kernel}.cu"));
        let ptx_file = out_dir.join(format!("{kernel}.ptx"));

        println!("cargo:rerun-if-changed={}", cu_file.display());

        let status = Command::new("nvcc")
            .args([
                "--ptx",
                &format!("-arch={nvcc_arch}"),
                "--use_fast_math",
                "--extra-device-vectorization",
                "-O3",
                "-o",
                ptx_file.to_str().unwrap(),
                cu_file.to_str().unwrap(),
            ])
            .status();

        match status {
            Ok(s) if s.success() => {
                println!(
                    "cargo:warning=compiled {kernel}.cu → {kernel}.ptx (arch={nvcc_arch})"
                );
            }
            Ok(s) => {
                // nvcc failed — create a placeholder PTX so the build can proceed
                // on machines without CUDA toolkit (e.g., CI).
                eprintln!("nvcc failed for {kernel}.cu with status {s}, creating stub PTX");
                std::fs::write(
                    &ptx_file,
                    format!("// STUB: {kernel}.cu was not compiled (nvcc unavailable)\n"),
                )
                .unwrap();
            }
            Err(e) => {
                eprintln!("nvcc not found ({e}), creating stub PTX for {kernel}.cu");
                std::fs::write(
                    &ptx_file,
                    format!("// STUB: {kernel}.cu was not compiled (nvcc not found)\n"),
                )
                .unwrap();
            }
        }
    }

    println!("cargo:rerun-if-env-changed=NORDOCR_CUDA_ARCH");
}
