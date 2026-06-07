use std::process::{Command, Stdio};
use std::thread;
use std::time::Instant;

fn run_concurrent_benchmarks(name: &str, command: &str, args: &[&str], count: usize) {
    println!(
        "\n=== Starting Concurrent Benchmark: {} ({} instances) ===",
        name, count
    );
    let start_time = Instant::now();
    let mut handles = vec![];

    for i in 0..count {
        let cmd_str = command.to_string();
        let args_vec: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let name_str = name.to_string();

        let handle = thread::spawn(move || {
            println!("[{}-{}] Starting...", name_str, i);
            let output = Command::new(&cmd_str)
                .args(&args_vec)
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .output()
                .expect("Failed to execute command");

            if !output.status.success() {
                eprintln!("[{}-{}] Failed with status: {}", name_str, i, output.status);
                // We should probably panic or signal failure, but for now just log
            } else {
                println!("[{}-{}] Completed successfully.", name_str, i);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let duration = start_time.elapsed();
    println!("=== Completed {} in {:.2?} ===", name, duration);
}

fn main() {
    // 1. Rust YOLOv8s Detect (3 instances)

    println!("Building Rust binary...");
    let build_status = Command::new("cargo")
        .args(&["build", "--release", "--bin", "performance_test"])
        .status()
        .expect("Failed to build");

    if !build_status.success() {
        eprintln!("Build failed!");
        return;
    }

    let rust_binary = "./target/release/performance_test";

    run_concurrent_benchmarks("Rust YOLOv8s Detect", rust_binary, &[], 3);

    // 2. PyTorch YOLOv8s Detect (3 instances)
    // python performance_test_mps.py --model yolov8s
    let python_interpreter = "/opt/anaconda3/envs/py3-14/bin/python";

    run_concurrent_benchmarks(
        "PyTorch YOLOv8s Detect",
        python_interpreter,
        &["performance_test_mps.py", "--model", "yolov8s"],
        3,
    );

    // 3. PyTorch YOLOv11 Drone (3 instances)
    // python performance_test_mps.py --model yolov11drone

    run_concurrent_benchmarks(
        "PyTorch YOLOv11 Drone",
        python_interpreter,
        &["performance_test_mps.py", "--model", "yolov11drone"],
        3,
    );

    println!("\nAll concurrent benchmarks completed.");
}
