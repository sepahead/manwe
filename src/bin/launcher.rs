use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result};
use clap::Parser;
use manwe::secure_io::resolve_executable;
use manwe::stream_url::{validate_rtsp_url, INVALID_RTSP_URL};

const CHILD_ENV_ALLOWLIST: &[&str] = &[
    "HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "SystemRoot",
    "WINDIR",
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "XDG_RUNTIME_DIR",
    "XDG_SESSION_TYPE",
    "DBUS_SESSION_BUS_ADDRESS",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "CUDA_VISIBLE_DEVICES",
    "WGPU_BACKEND",
    "WGPU_POWER_PREF",
    "RUST_LOG",
    "RUST_BACKTRACE",
];

fn apply_child_environment(command: &mut Command, urls: &[String]) {
    command.env_clear();
    for name in CHILD_ENV_ALLOWLIST {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    command.env("MANWE_RTSP_URLS", urls.join("\u{1f}"));
}

fn sha256_digest(value: &str) -> std::result::Result<String, String> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(value.to_ascii_lowercase())
    } else {
        Err("SHA-256 must contain exactly 64 hexadecimal characters".to_string())
    }
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// RTSP/video URLs. Prefer MANWE_RTSP_URLS for values containing credentials.
    #[arg(
        long = "url",
        env = "MANWE_RTSP_URLS",
        hide_env_values = true,
        value_delimiter = '\x1f',
        required = true,
        num_args = 1..
    )]
    urls: Vec<String>,

    /// Override the camera_view executable.
    #[arg(long)]
    camera_view: Option<PathBuf>,

    /// Forward CPU-only inference to camera_view.
    #[arg(long)]
    cpu: bool,

    /// Forward an ffmpeg executable/path to camera_view.
    #[arg(long, env = "MANWE_FFMPEG", hide_env_values = true)]
    ffmpeg: Option<PathBuf>,

    /// Forward a local safetensors model to camera_view.
    #[arg(long, env = "MANWE_MODEL", hide_env_values = true)]
    model: PathBuf,

    /// Forward the expected SHA-256 for the model artifact.
    #[arg(
        long,
        env = "MANWE_MODEL_SHA256",
        value_parser = sha256_digest,
        hide_env_values = true
    )]
    model_sha256: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.urls.is_empty() || args.urls.iter().any(|url| validate_rtsp_url(url).is_err()) {
        anyhow::bail!(INVALID_RTSP_URL)
    }
    let executable = match args.camera_view {
        Some(path) => path,
        None => std::env::current_exe()
            .context("failed to locate launcher executable")?
            .with_file_name("camera_view"),
    };
    let executable = resolve_executable(&executable)?;
    let ffmpeg = resolve_executable(
        args.ffmpeg
            .as_deref()
            .unwrap_or_else(|| std::path::Path::new("ffmpeg")),
    )?;

    executable.verify()?;
    ffmpeg.verify()?;
    let mut command = Command::new(executable.path());
    // Keep credential-bearing URLs out of the child command line.
    apply_child_environment(&mut command, &args.urls);
    if args.cpu {
        command.arg("--cpu");
    }
    command.arg("--ffmpeg").arg(ffmpeg.path());
    command.arg("--model").arg(args.model);
    command.arg("--model-sha256").arg(args.model_sha256);

    let status = command
        .status()
        .with_context(|| format!("failed to start {}", executable.path().display()))?;
    if !status.success() {
        anyhow::bail!("camera_view exited with {status}")
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launcher_rejects_control_characters_and_bad_digests() {
        assert!(validate_rtsp_url("rtsp://example.invalid/live\nnext").is_err());
        assert!(sha256_digest("not-a-digest").is_err());
        assert!(sha256_digest(&"0".repeat(64)).is_ok());
    }

    #[test]
    fn child_environment_is_an_explicit_allowlist_plus_private_urls() {
        let mut command = Command::new("unused");
        apply_child_environment(&mut command, &["rtsp://example.invalid/live".to_string()]);

        for (name, _) in command.get_envs() {
            let name = name.to_string_lossy();
            assert!(name == "MANWE_RTSP_URLS" || CHILD_ENV_ALLOWLIST.contains(&name.as_ref()));
        }
    }
}
