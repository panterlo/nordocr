//! Integration test: verify TRT FFI loads a real engine file.

use std::path::Path;

#[test]
fn test_load_svtrv2_engine() {
    let engine_path = Path::new("/home/sysop/nordocr/models/recognize_svtrv2_sm120.engine");
    if !engine_path.exists() {
        eprintln!("Skipping: engine file not found");
        return;
    }

    // Create runtime (calls trt_create_runtime via FFI).
    let runtime = nordocr_trt::TrtRuntime::new().expect("failed to create TRT runtime");

    // Load engine (calls trt_deserialize_engine via FFI).
    let engine =
        nordocr_trt::TrtEngine::load(&runtime, engine_path).expect("failed to load engine");

    // Verify I/O tensor names.
    let inputs = engine.input_names();
    let outputs = engine.output_names();
    assert_eq!(inputs.len(), 1, "SVTRv2 should have 1 input");
    assert_eq!(outputs.len(), 1, "SVTRv2 should have 1 output");
    assert_eq!(inputs[0], "input");
    assert_eq!(outputs[0], "output");

    eprintln!("Engine loaded successfully!");
    eprintln!("  Inputs: {:?}", inputs);
    eprintln!("  Outputs: {:?}", outputs);
    eprintln!("  Input dtypes: {:?}", engine.input_dtypes());
    eprintln!("  Output dtypes: {:?}", engine.output_dtypes());

    // Create execution context (calls trt_create_execution_context via FFI).
    let _ctx = engine.create_context().expect("failed to create context");
    eprintln!("  Execution context created successfully!");
}
