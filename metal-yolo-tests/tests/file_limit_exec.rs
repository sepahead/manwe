#![cfg(unix)]

use std::fs;
use std::process::{Command, Stdio};

use manwe::secure_io::resolve_executable;

const FILE_LIMIT_EXEC_MODE: &str = "--manwe-internal-file-limit-exec";

#[test]
fn production_wrapper_enforces_the_child_file_size_limit() {
    let directory = std::env::temp_dir().join(format!(
        "manwe-file-limit-exec-integration-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir(&directory).unwrap();
    let output = directory.join("bounded-output.bin");
    let limit = 4096_u64;
    let test_executable = resolve_executable(&std::env::current_exe().unwrap()).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_benchmark_video"));
    command
        .arg(FILE_LIMIT_EXEC_MODE)
        .arg(limit.to_string())
        .arg(test_executable.sha256())
        .arg(test_executable.path())
        .args(["--exact", "file_size_limit_exec_helper"])
        .env_clear()
        .env("MANWE_FILE_SIZE_LIMIT_EXEC_OUTPUT", &output)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    let status = command.status().unwrap();
    let observed_size = fs::metadata(&output).unwrap().len();

    assert!(!status.success() && observed_size <= limit);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn production_wrapper_never_executes_a_digest_mismatched_replacement() {
    let directory = std::env::temp_dir().join(format!(
        "manwe-file-limit-digest-integration-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir(&directory).unwrap();
    let target = directory.join("target");
    fs::copy(std::env::current_exe().unwrap(), &target).unwrap();
    let authenticated = resolve_executable(&target).unwrap();
    let expected_sha256 = authenticated.sha256().to_string();
    drop(authenticated);
    fs::remove_file(&target).unwrap();
    fs::copy("/bin/echo", &target).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_benchmark_video"));
    command
        .arg(FILE_LIMIT_EXEC_MODE)
        .arg("4096")
        .arg(expected_sha256)
        .arg(&target)
        .arg("replacement-must-not-run")
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    let status = command.status().unwrap();

    assert!(!status.success());
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn file_size_limit_exec_helper() {
    let Some(output) = std::env::var_os("MANWE_FILE_SIZE_LIMIT_EXEC_OUTPUT") else {
        return;
    };
    fs::write(output, vec![0_u8; 65_536]).unwrap();
}
