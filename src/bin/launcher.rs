use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result};
use clap::Parser;
use manwe::secure_io::resolve_executable;
use manwe::stream_url::{validate_rtsp_url, INVALID_RTSP_URL, MAX_STREAMS};

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

    /// Forward a schema-2 model contract to camera_view.
    #[arg(long, env = "MANWE_CONTRACT", hide_env_values = true)]
    contract: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    // Clap has copied the values into owned arguments. Remove every launcher-
    // specific ambient input before executable discovery or process creation;
    // the child receives only the explicit, allowlisted values below.
    for name in ["MANWE_RTSP_URLS", "MANWE_FFMPEG", "MANWE_CONTRACT"] {
        std::env::remove_var(name);
    }
    if args.urls.is_empty() || args.urls.iter().any(|url| validate_rtsp_url(url).is_err()) {
        anyhow::bail!(INVALID_RTSP_URL)
    }
    if args.urls.len() > MAX_STREAMS {
        anyhow::bail!("at most {MAX_STREAMS} concurrent streams are supported")
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

    executable.require_native_executable()?;
    ffmpeg.require_native_executable()?;
    executable.verify()?;
    ffmpeg.verify()?;
    let mut command = Command::new(executable.path());
    // Keep credential-bearing URLs out of the child command line.
    apply_child_environment(&mut command, &args.urls);
    if args.cpu {
        command.arg("--cpu");
    }
    command.arg("--ffmpeg").arg(ffmpeg.path());
    command.arg("--contract").arg(args.contract);

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
    fn launcher_rejects_control_characters_in_stream_urls() {
        assert!(validate_rtsp_url("rtsp://example.invalid/live\nnext").is_err());
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
