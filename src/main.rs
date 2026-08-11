use std::collections::HashSet;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use clap::Parser;
use image::DynamicImage;
use manwe::native_runtime::{NativeInferenceReport, NativeRuntime};
use manwe::runtime_contract::ModelTask;
use manwe::secure_io::{
    bounded_open_file_identity, open_bounded_regular_file, read_bounded_regular_file, sha256_hex,
    BoundDirectory, FileIdentity, HardLinkPublication, MAX_ENCODED_IMAGE_BYTES,
};
use manwe::{DetectionRecord, PoseRecord};
use serde::Serialize;

const MAX_OUTPUT_ATTEMPTS: u32 = 10_000;
const MAX_OUTPUT_JPEG_BYTES: u64 = 256 * 1024 * 1024;
const MAX_TRACE_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_INPUT_IMAGES: usize = 10_000;
const STAGED_OUTPUT_NAME: &str = "output.jpg";

fn secure_name_token(purpose: &str) -> Result<String> {
    let mut entropy = [0_u8; 32];
    getrandom::fill(&mut entropy)
        .with_context(|| format!("failed to obtain OS randomness for {purpose}"))?;
    Ok(sha256_hex(&entropy))
}

fn has_io_error_kind(error: &anyhow::Error, kind: std::io::ErrorKind) -> bool {
    error
        .chain()
        .filter_map(|cause| cause.downcast_ref::<std::io::Error>())
        .any(|cause| cause.kind() == kind)
}

fn bounded_legend_size(value: &str) -> std::result::Result<u32, String> {
    let parsed = value
        .parse::<u32>()
        .map_err(|_| format!("{value:?} is not a non-negative integer"))?;
    if parsed <= 256 {
        Ok(parsed)
    } else {
        Err("legend size must not exceed 256 pixels".to_string())
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than an enabled Metal/CUDA backend.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace JSON file).
    #[arg(long)]
    tracing: bool,

    /// Schema-2 contract beside the exact model artifact.
    #[arg(long, env = "MANWE_CONTRACT")]
    contract: PathBuf,

    /// One or more input images.
    #[arg(required = true, num_args = 1..=10_000)]
    images: Vec<PathBuf>,

    /// Existing owner-controlled directory for annotated JPEGs.
    /// Defaults to each input image's directory.
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// Legend size; zero disables labels.
    #[arg(long, default_value_t = 14, value_parser = bounded_legend_size)]
    legend_size: u32,
}

#[derive(Serialize)]
struct InferenceReceipt<'a> {
    schema_version: &'static str,
    task: &'static str,
    device: &'static str,
    model_name: &'a str,
    model_version: &'a str,
    contract_path: &'a str,
    contract_sha256: &'a str,
    artifact_path: &'a str,
    artifact_sha256: &'a str,
    input_path: &'a str,
    input_sha256: &'a str,
    output_path: &'a str,
    output_sha256: &'a str,
    objects: &'a InferenceObjects,
}

#[derive(Serialize)]
#[serde(tag = "kind", content = "items", rename_all = "snake_case")]
enum InferenceObjects {
    Detections(Vec<DetectionRecord>),
    Poses(Vec<PoseRecord>),
}

fn create_trace_file(parent: &Path) -> Result<(fs::File, PathBuf, BoundDirectory)> {
    let parent = BoundDirectory::open(parent).context("failed to bind trace output directory")?;
    parent.require_owner_mutation_boundary()?;
    for _ in 0..MAX_OUTPUT_ATTEMPTS {
        let token = secure_name_token("trace-file naming")?;
        let name = OsString::from(format!("manwe-trace-{token}.json"));
        let path = parent.path().join(&name);
        parent.verify()?;
        match parent.create_new_regular_file_entry(&name, 0o600) {
            Ok(file) => {
                parent.sync().with_context(|| {
                    format!(
                        "trace file was created at {}, but its directory entry has unknown durability",
                        path.display()
                    )
                })?;
                return Ok((file, path, parent));
            }
            Err(error) if has_io_error_kind(&error, std::io::ErrorKind::AlreadyExists) => continue,
            Err(error) => return Err(error).context("failed to create private trace output"),
        }
    }
    anyhow::bail!("could not reserve a unique trace output name")
}

fn remember_trace_write_error(error: &Mutex<Option<String>>, message: String) {
    let mut slot = error
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if slot.is_none() {
        *slot = Some(message);
    }
}

struct CheckedTraceWriter {
    file: Arc<Mutex<fs::File>>,
    write_error: Arc<Mutex<Option<String>>>,
}

impl Write for CheckedTraceWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let result = self
            .file
            .lock()
            .map_err(|_| std::io::Error::other("trace file lock was poisoned"))
            .and_then(|mut file| {
                let current = file.metadata()?.len();
                let incoming = u64::try_from(bytes.len())
                    .map_err(|_| std::io::Error::other("trace write length overflowed"))?;
                if incoming > MAX_TRACE_BYTES.saturating_sub(current) {
                    return Err(std::io::Error::other(format!(
                        "trace exceeds the {MAX_TRACE_BYTES}-byte limit"
                    )));
                }
                file.write(bytes)
            });
        match result {
            Ok(written) => Ok(written),
            Err(error) => {
                remember_trace_write_error(&self.write_error, error.to_string());
                // The dependency unwraps writer errors on its background thread.
                // Record the first failure and act as a sink so the foreground can
                // join the thread and return a normal, contextualized error.
                Ok(bytes.len())
            }
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let result = self
            .file
            .lock()
            .map_err(|_| std::io::Error::other("trace file lock was poisoned"))
            .and_then(|mut file| file.flush());
        if let Err(error) = result {
            remember_trace_write_error(&self.write_error, error.to_string());
        }
        Ok(())
    }
}

struct TraceCompletion {
    path: PathBuf,
    parent: BoundDirectory,
    file: Arc<Mutex<fs::File>>,
    write_error: Arc<Mutex<Option<String>>>,
}

impl TraceCompletion {
    fn finish(self) -> Result<()> {
        let write_error = self
            .write_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(error) = write_error {
            anyhow::bail!("failed to write trace {}: {error}", self.path.display())
        }
        let name = self
            .path
            .file_name()
            .context("trace output path lost its filename")?;
        let written_identity = {
            let file = self
                .file
                .lock()
                .map_err(|_| anyhow::anyhow!("trace file lock was poisoned"))?;
            file.sync_all()
                .with_context(|| format!("failed to synchronize trace {}", self.path.display()))?;
            bounded_open_file_identity(&file, &self.path, MAX_TRACE_BYTES)?
        };
        let visible_identity = self
            .parent
            .regular_file_entry_identity(name, MAX_TRACE_BYTES)
            .with_context(|| {
                format!(
                    "trace bytes were synchronized, but {} no longer identifies them",
                    self.path.display()
                )
            })?;
        if visible_identity != written_identity {
            anyhow::bail!(
                "trace bytes were synchronized, but {} was replaced before completion",
                self.path.display()
            )
        }
        self.parent.sync().with_context(|| {
            format!(
                "trace data was synchronized at {}, but its directory entry has unknown durability",
                self.path.display()
            )
        })
    }
}

struct TraceSession {
    guard: tracing_chrome::FlushGuard,
    completion: TraceCompletion,
}

impl TraceSession {
    fn finish(self) -> Result<()> {
        let Self { guard, completion } = self;
        drop(guard);
        completion.finish()
    }
}

fn verify_jpeg_file_at(
    directory: &BoundDirectory,
    name: &OsStr,
    expected: &[u8],
) -> Result<FileIdentity> {
    if expected.is_empty() || expected.len() as u64 > MAX_OUTPUT_JPEG_BYTES {
        anyhow::bail!("annotated JPEG must contain between 1 and {MAX_OUTPUT_JPEG_BYTES} bytes")
    }
    let path = directory.path().join(name);
    let (actual, identity) =
        directory.read_bounded_regular_file_entry_with_identity(name, MAX_OUTPUT_JPEG_BYTES)?;
    if actual != expected {
        anyhow::bail!("annotated JPEG verification failed: {}", path.display())
    }
    let mut reader =
        image::ImageReader::with_format(Cursor::new(&actual), image::ImageFormat::Jpeg);
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(32_768);
    limits.max_image_height = Some(32_768);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    reader
        .decode()
        .with_context(|| format!("staged output is not a valid JPEG: {}", path.display()))?;
    Ok(identity)
}

fn write_verified_jpeg_once(
    directory: &BoundDirectory,
    name: &OsStr,
    encoded: &[u8],
) -> Result<()> {
    let path = directory.path().join(name);
    // The hard link published later exposes this exact inode. Set the mode at
    // descriptor-relative creation so neither a pathname swap nor a permissive
    // process umask can redirect or broaden the output authority.
    let mut file = directory.create_new_regular_file_entry(name, 0o600)?;
    file.write_all(encoded)?;
    file.sync_all()?;
    let written_identity = bounded_open_file_identity(&file, &path, MAX_OUTPUT_JPEG_BYTES)?;
    drop(file);
    let visible_identity = verify_jpeg_file_at(directory, name, encoded)?;
    if visible_identity != written_identity {
        anyhow::bail!("staged JPEG was replaced after its synchronized write")
    }
    Ok(())
}

struct ImagePublication {
    parent_dir: BoundDirectory,
    stage_dir: BoundDirectory,
    stage_name: OsString,
    final_link_created: bool,
    committed: bool,
    cleanup_started: bool,
}

impl ImagePublication {
    fn acquire(parent_dir: &BoundDirectory) -> Result<Self> {
        let parent_dir = parent_dir.try_clone()?;
        parent_dir.require_owner_mutation_boundary()?;
        for _ in 0..MAX_OUTPUT_ATTEMPTS {
            let token = secure_name_token("image-output staging")?;
            let stage_name = OsString::from(format!(".manwe-image-output-{token}.in-progress"));
            parent_dir.verify()?;
            match parent_dir.create_directory_entry(&stage_name, 0o700) {
                Ok(stage_dir) => {
                    let publication = Self {
                        parent_dir,
                        stage_dir,
                        stage_name,
                        final_link_created: false,
                        committed: false,
                        cleanup_started: false,
                    };
                    publication.parent_dir.sync().with_context(|| {
                        format!(
                            "image-output staging marker was created at {}, but its visibility or durability is unknown",
                            publication.stage_dir.path().display()
                        )
                    })?;
                    publication
                        .stage_dir
                        .require_owner_mutation_boundary()
                        .with_context(|| {
                            format!(
                                "image-output staging marker was created at {}, but could not be authenticated",
                                publication.stage_dir.path().display()
                            )
                        })?;
                    return Ok(publication);
                }
                Err(error) if has_io_error_kind(&error, std::io::ErrorKind::AlreadyExists) => {
                    continue;
                }
                Err(error) => return Err(error),
            }
        }
        anyhow::bail!("could not reserve a unique image-output staging directory")
    }

    fn publish(&mut self, encoded: &[u8], base: &OsStr) -> Result<PathBuf> {
        self.publish_with_hooks(encoded, base, |_| Ok(()), |_| Ok(()))
    }

    fn publish_with_hooks<AfterLink, BeforeCommit>(
        &mut self,
        encoded: &[u8],
        base: &OsStr,
        mut after_link: AfterLink,
        before_commit: BeforeCommit,
    ) -> Result<PathBuf>
    where
        AfterLink: FnMut(&Path) -> Result<()>,
        BeforeCommit: FnOnce(&Path) -> Result<()>,
    {
        self.parent_dir.require_owner_mutation_boundary()?;
        self.stage_dir.require_owner_mutation_boundary()?;
        let staged_output_name = OsStr::new(STAGED_OUTPUT_NAME);
        write_verified_jpeg_once(&self.stage_dir, staged_output_name, encoded)?;
        self.stage_dir.sync()?;
        let mut output = None;
        for attempt in 0..MAX_OUTPUT_ATTEMPTS {
            let mut name = OsString::from(base);
            if attempt > 0 {
                name.push(format!(".{attempt}"));
            }
            name.push(".jpg");
            let candidate = self.parent_dir.path().join(&name);
            self.parent_dir.verify()?;
            self.stage_dir.verify()?;
            match self.stage_dir.hard_link_file_entry_to(
                staged_output_name,
                &self.parent_dir,
                &name,
                MAX_OUTPUT_JPEG_BYTES,
            ) {
                Ok(HardLinkPublication::Created | HardLinkPublication::AlreadyLinked) => {
                    self.final_link_created = true;
                    let mut authenticate = || -> Result<()> {
                        after_link(&candidate)?;
                        let stage_identity =
                            verify_jpeg_file_at(&self.stage_dir, staged_output_name, encoded)?;
                        self.parent_dir.verify()?;
                        self.stage_dir.verify()?;
                        let output_identity =
                            verify_jpeg_file_at(&self.parent_dir, &name, encoded)?;
                        if output_identity != stage_identity {
                            anyhow::bail!(
                                "published JPEG identity does not match the staged output"
                            )
                        }
                        Ok(())
                    };
                    authenticate().with_context(|| {
                        format!(
                            "a final link was created at {}, but its content or visibility could not be authenticated",
                            candidate.display()
                        )
                    })?;
                    output = Some(candidate);
                    break;
                }
                Ok(HardLinkPublication::DestinationOccupied) => continue,
                Err(error) => {
                    // A remote filesystem may publish a hard link and then
                    // report failure. Preserve the private stage and identify
                    // the exact candidate instead of pretending no side effect
                    // occurred or retrying under another name.
                    self.final_link_created = true;
                    return Err(error).with_context(|| {
                        format!(
                            "hard-link publication at {} failed with an indeterminate visibility state",
                            candidate.display()
                        )
                    });
                }
            }
        }
        let output = output.context("could not find an unused image-output name")?;
        before_commit(&output).with_context(|| {
            format!(
                "the final link last authenticated at {} has unknown publication durability",
                output.display()
            )
        })?;
        self.parent_dir.sync().with_context(|| {
            format!(
                "the final link last authenticated at {} has unknown publication durability",
                output.display()
            )
        })?;
        self.committed = true;
        self.cleanup_started = true;
        self.cleanup_staging(true).with_context(|| {
            format!(
                "output is committed at {}, but exact-entry staging cleanup is incomplete or its durability is unknown",
                output.display()
            )
        })?;
        Ok(output)
    }

    fn cleanup_staging(&self, require_staged_output: bool) -> Result<()> {
        self.cleanup_staging_with_hook(require_staged_output, || {})
    }

    fn cleanup_staging_with_hook<F>(
        &self,
        require_staged_output: bool,
        after_directory_removal: F,
    ) -> Result<()>
    where
        F: FnOnce(),
    {
        self.parent_dir.require_owner_mutation_boundary()?;
        self.stage_dir.require_owner_mutation_boundary()?;

        let output_name = OsStr::new(STAGED_OUTPUT_NAME);
        if require_staged_output {
            self.stage_dir.remove_file_entry(output_name)?;
        } else {
            self.stage_dir.remove_file_entry_if_exists(output_name)?;
        }
        self.stage_dir.sync()?;

        self.parent_dir.remove_directory_entry(&self.stage_name)?;
        after_directory_removal();
        self.parent_dir.sync()
    }
}

impl Drop for ImagePublication {
    fn drop(&mut self) {
        if !self.cleanup_started && !self.committed && !self.final_link_created {
            self.cleanup_started = true;
            let _ = self.cleanup_staging(false);
        }
    }
}

fn save_output_with_digest(
    input: &Path,
    image: &DynamicImage,
    output_dir: &BoundDirectory,
) -> Result<(PathBuf, String)> {
    let stem = input.file_stem().context("input image needs a file name")?;
    let mut base = stem.to_os_string();
    base.push(".pp");
    let mut encoded = Cursor::new(Vec::new());
    image
        .write_to(&mut encoded, image::ImageFormat::Jpeg)
        .context("failed to encode annotated JPEG")?;
    if encoded.get_ref().len() as u64 > MAX_OUTPUT_JPEG_BYTES {
        anyhow::bail!("annotated JPEG exceeds the {MAX_OUTPUT_JPEG_BYTES}-byte limit")
    }
    let digest = sha256_hex(encoded.get_ref());
    let mut publication = ImagePublication::acquire(output_dir)?;
    let path = publication.publish(encoded.get_ref(), &base)?;
    Ok((path, digest))
}

#[cfg(test)]
fn save_output(input: &Path, image: &DynamicImage) -> Result<PathBuf> {
    let parent = input
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = BoundDirectory::open(parent)?;
    save_output_with_digest(input, image, &parent).map(|(path, _digest)| path)
}

fn decode_image(path: &Path) -> Result<(DynamicImage, String)> {
    let bytes = read_bounded_regular_file(path, MAX_ENCODED_IMAGE_BYTES)?;
    let digest = sha256_hex(&bytes);
    let mut reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .with_context(|| format!("failed to determine image format for {}", path.display()))?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(32_768);
    limits.max_image_height = Some(32_768);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let image = reader
        .decode()
        .with_context(|| format!("failed to decode image {}", path.display()))?;
    Ok((image, digest))
}

fn utf8_path(path: &Path) -> Result<&str> {
    path.to_str().with_context(|| {
        format!(
            "path must be valid UTF-8 for structured output: {}",
            path.display()
        )
    })
}

fn canonical_input_path(path: &Path) -> Result<PathBuf> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .context("input path must identify one image file")?;
    let parent = BoundDirectory::open(parent).context("failed to bind input image directory")?;
    let result = parent.path().join(name);
    parent.verify()?;
    Ok(result)
}

fn preflight_image_paths(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    if paths.is_empty() || paths.len() > MAX_INPUT_IMAGES {
        anyhow::bail!("input batch must contain between 1 and {MAX_INPUT_IMAGES} images")
    }
    let mut seen = HashSet::with_capacity(paths.len());
    let mut canonical_paths = Vec::with_capacity(paths.len());
    for path in paths {
        let canonical = canonical_input_path(path)?;
        utf8_path(&canonical)?;
        if !seen.insert(canonical.clone()) {
            anyhow::bail!("input image paths must not contain duplicates")
        }
        let _ =
            open_bounded_regular_file(&canonical, MAX_ENCODED_IMAGE_BYTES).with_context(|| {
                format!(
                    "input image failed bounded preflight: {}",
                    canonical.display()
                )
            })?;
        canonical_paths.push(canonical);
    }
    Ok(canonical_paths)
}

fn preflight_default_output_directories(image_paths: &[PathBuf]) -> Result<()> {
    let mut seen = HashSet::new();
    for image_path in image_paths {
        let parent = image_path
            .parent()
            .context("canonical input image must have a parent directory")?;
        if seen.insert(parent.to_path_buf()) {
            let directory = BoundDirectory::open(parent)
                .context("failed to bind default annotated-image output directory")?;
            directory.require_owner_mutation_boundary()?;
            utf8_path(directory.path())?;
        }
    }
    Ok(())
}

fn run(args: Args) -> Result<()> {
    // Prove all caller-controlled filesystem authority before allocating a
    // model or initializing an accelerator. Each image is reopened and fully
    // authenticated at consumption time; this pass rejects impossible batches,
    // duplicate paths, missing/special inputs, and oversized files without
    // expensive model side effects.
    let image_paths = preflight_image_paths(&args.images)?;
    let output_dir = match args.output_dir {
        Some(path) => {
            let directory = BoundDirectory::open(&path)
                .context("failed to bind annotated-image output directory")?;
            directory.require_owner_mutation_boundary()?;
            directory.verify()?;
            utf8_path(directory.path())?;
            Some(directory)
        }
        None => {
            preflight_default_output_directories(&image_paths)?;
            None
        }
    };
    let runtime = NativeRuntime::load(&args.contract, args.cpu)?;
    let contract = runtime.contract();
    let contract_path = utf8_path(runtime.contract_path())?;
    let artifact_path = utf8_path(runtime.artifact_path())?;

    for image_path in &image_paths {
        // Retain the exact default output-directory descriptor across decoding,
        // inference, and publication. A dedicated directory is already retained
        // for the whole batch.
        let per_image_output_dir = if output_dir.is_none() {
            let parent = image_path
                .parent()
                .context("canonical input image must have a parent directory")?;
            let directory = BoundDirectory::open(parent)
                .context("failed to rebind default annotated-image output directory")?;
            directory.require_owner_mutation_boundary()?;
            Some(directory)
        } else {
            None
        };
        let output_directory = output_dir
            .as_ref()
            .or(per_image_output_dir.as_ref())
            .context("validated output directory was not retained")?;
        output_directory.verify()?;
        let (original, input_sha256) = decode_image(image_path)?;
        let (annotated, objects) = match runtime.infer(original, args.legend_size)? {
            NativeInferenceReport::Detect(report) => (
                report.annotated_image,
                InferenceObjects::Detections(report.detections),
            ),
            NativeInferenceReport::Pose(report) => (
                report.annotated_image,
                InferenceObjects::Poses(report.poses),
            ),
        };
        let (output_path, output_sha256) =
            save_output_with_digest(image_path, &annotated, output_directory)?;
        output_directory.verify().with_context(|| {
            format!(
                "output is committed at the last-authenticated path {}, but its parent directory identity changed before receipt emission",
                output_path.display()
            )
        })?;
        let committed_output_path = utf8_path(&output_path)?;
        let receipt = InferenceReceipt {
            schema_version: "manwe-inference-result-v1",
            task: match contract.task() {
                ModelTask::Detect => "detect",
                ModelTask::Pose => "pose",
            },
            device: runtime.device_name(),
            model_name: contract.model_name(),
            model_version: contract.model_version(),
            contract_path,
            contract_sha256: runtime.contract_sha256(),
            artifact_path,
            artifact_sha256: contract.artifact_sha256(),
            input_path: utf8_path(image_path)?,
            input_sha256: &input_sha256,
            output_path: committed_output_path,
            output_sha256: &output_sha256,
            objects: &objects,
        };
        let stdout = std::io::stdout();
        let mut output = stdout.lock();
        serde_json::to_writer(&mut output, &receipt).with_context(|| {
            format!(
                "output is committed at {committed_output_path}, but its inference receipt \
                     could not be serialized to stdout"
            )
        })?;
        output.write_all(b"\n").with_context(|| {
            format!(
                "output is committed at {committed_output_path}, but its inference receipt \
                 newline could not be written to stdout"
            )
        })?;
        output.flush().with_context(|| {
            format!(
                "output is committed at {committed_output_path}, but its inference receipt \
                     could not be flushed to stdout"
            )
        })?;
    }

    Ok(())
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let trace_session = if args.tracing {
        let (trace_file, trace_path, trace_parent) = create_trace_file(Path::new("."))?;
        let trace_file = Arc::new(Mutex::new(trace_file));
        let write_error = Arc::new(Mutex::new(None));
        let trace_writer = CheckedTraceWriter {
            file: Arc::clone(&trace_file),
            write_error: Arc::clone(&write_error),
        };
        eprintln!("writing private trace to {}", trace_path.display());
        // Supplying our own exclusive, owner-private writer avoids the
        // dependency's panic-prone, truncating default File::create path.
        let (chrome_layer, guard) = ChromeLayerBuilder::new().writer(trace_writer).build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(TraceSession {
            guard,
            completion: TraceCompletion {
                path: trace_path,
                parent: trace_parent,
                file: trace_file,
                write_error,
            },
        })
    } else {
        None
    };

    let inference = run(args);
    let trace = trace_session.map_or(Ok(()), TraceSession::finish);
    match (inference, trace) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(error)) => Err(error),
        (Err(error), Err(trace_error)) => {
            Err(error.context(format!("trace finalization also failed: {trace_error:#}")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn acquire_publication(path: &Path) -> Result<ImagePublication> {
        let directory = BoundDirectory::open(path)?;
        ImagePublication::acquire(&directory)
    }

    fn encoded_test_jpeg() -> Vec<u8> {
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(2, 2));
        let mut encoded = Cursor::new(Vec::new());
        image
            .write_to(&mut encoded, image::ImageFormat::Jpeg)
            .unwrap();
        encoded.into_inner()
    }

    #[test]
    fn cli_requires_one_contract_instead_of_independent_model_guesses() {
        assert!(Args::try_parse_from(["manwe", "frame.jpg"]).is_err());
        let args =
            Args::try_parse_from(["manwe", "--contract", "model.contract.json", "frame.jpg"])
                .unwrap();
        assert_eq!(args.contract, PathBuf::from("model.contract.json"));
        assert_eq!(args.images, [PathBuf::from("frame.jpg")]);
        assert_eq!(args.output_dir, None);
    }

    #[test]
    fn private_entry_tokens_have_full_lowercase_sha256_shape() {
        let first = secure_name_token("test entry").unwrap();
        let second = secure_name_token("test entry").unwrap();

        assert_eq!(first.len(), 64);
        assert!(first
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
        assert_ne!(first, second);
    }

    #[test]
    fn cli_accepts_a_dedicated_output_directory() {
        let args = Args::try_parse_from([
            "manwe",
            "--contract",
            "model.contract.json",
            "--output-dir",
            "annotated",
            "frame.jpg",
        ])
        .unwrap();

        assert_eq!(args.output_dir, Some(PathBuf::from("annotated")));
        assert_eq!(args.images, [PathBuf::from("frame.jpg")]);
    }

    #[test]
    fn cli_bounds_the_input_batch_before_runtime_construction() {
        let mut arguments = vec![
            OsString::from("manwe"),
            OsString::from("--contract"),
            OsString::from("model.contract.json"),
        ];
        arguments
            .extend((0..=MAX_INPUT_IMAGES).map(|index| OsString::from(format!("{index}.jpg"))));

        assert!(Args::try_parse_from(arguments).is_err());
    }

    #[test]
    fn input_preflight_rejects_duplicate_and_missing_paths() {
        let directory = std::env::temp_dir().join(format!(
            "manwe-input-preflight-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let input = directory.join("frame.jpg");
        fs::write(&input, b"not decoded during preflight").unwrap();

        let duplicate = preflight_image_paths(&[input.clone(), input.clone()]).unwrap_err();
        assert!(duplicate.to_string().contains("duplicates"));
        let missing = preflight_image_paths(&[directory.join("missing.jpg")]).unwrap_err();
        assert!(missing.to_string().contains("bounded preflight"));

        fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn default_output_preflight_rejects_a_shared_mutation_boundary() {
        use std::os::unix::fs::PermissionsExt;

        let directory = std::env::temp_dir().join(format!(
            "manwe-default-output-preflight-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let input = directory.join("frame.jpg");
        fs::write(&input, b"not decoded during preflight").unwrap();
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o777)).unwrap();

        let error = preflight_default_output_directories(&[input]).unwrap_err();
        assert!(error.to_string().contains("group- or world-writable"));

        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700)).unwrap();
        fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn trace_creation_is_exclusive_and_owner_private() {
        use std::os::unix::fs::PermissionsExt;

        let directory = std::env::temp_dir().join(format!(
            "manwe-trace-output-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();

        let (mut file, path, parent) = create_trace_file(&directory).unwrap();
        file.write_all(b"[]").unwrap();
        file.sync_all().unwrap();
        assert_eq!(fs::metadata(&path).unwrap().permissions().mode() & 0o077, 0);
        assert!(path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("manwe-trace-"));
        let (_second_file, second_path, second_parent) = create_trace_file(&directory).unwrap();
        assert_ne!(path, second_path);
        assert_eq!(fs::read(&path).unwrap(), b"[]");

        drop(file);
        drop(parent);
        drop(second_parent);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn trace_writer_enforces_its_limit_before_writing() {
        let directory = std::env::temp_dir().join(format!(
            "manwe-trace-limit-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let (file, path, parent) = create_trace_file(&directory).unwrap();
        file.set_len(MAX_TRACE_BYTES).unwrap();
        let file = Arc::new(Mutex::new(file));
        let write_error = Arc::new(Mutex::new(None));
        let mut writer = CheckedTraceWriter {
            file: Arc::clone(&file),
            write_error: Arc::clone(&write_error),
        };

        assert_eq!(writer.write(b"x").unwrap(), 1);
        assert_eq!(fs::metadata(&path).unwrap().len(), MAX_TRACE_BYTES);
        assert!(write_error
            .lock()
            .unwrap()
            .as_deref()
            .is_some_and(|error| error.contains("trace exceeds")));

        drop(writer);
        drop(file);
        drop(parent);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn inference_receipt_has_exactly_one_tagged_object_payload() {
        let objects = InferenceObjects::Detections(vec![DetectionRecord {
            prediction_index: 7,
            class_index: 1,
            class_name: "bird".to_owned(),
            airspace_class: Some("bird".to_owned()),
            confidence: 0.75,
            bbox_xyxy: [1.0, 2.0, 3.0, 4.0],
        }]);
        let receipt = InferenceReceipt {
            schema_version: "manwe-inference-result-v1",
            task: "detect",
            device: "cpu",
            model_name: "model",
            model_version: "1",
            contract_path: "/contract.json",
            contract_sha256: "contract-digest",
            artifact_path: "/model.safetensors",
            artifact_sha256: "artifact-digest",
            input_path: "/frame.png",
            input_sha256: "input-digest",
            output_path: "/frame.pp.jpg",
            output_sha256: "output-digest",
            objects: &objects,
        };

        let value = serde_json::to_value(receipt).unwrap();
        assert_eq!(value["objects"]["kind"], "detections");
        assert_eq!(value["objects"]["items"][0]["prediction_index"], 7);
        assert_eq!(value["objects"]["items"][0]["airspace_class"], "bird");
        assert!(value.get("detections").is_none());
        assert!(value.get("poses").is_none());
    }

    #[test]
    fn output_creation_never_overwrites_an_existing_candidate() {
        let directory = std::env::temp_dir().join(format!(
            "manwe-output-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let input = directory.join("frame.png");
        std::fs::write(&input, b"input-marker").unwrap();
        let occupied = directory.join("frame.pp.jpg");
        std::fs::write(&occupied, b"do-not-overwrite").unwrap();

        let image = DynamicImage::ImageRgb8(image::RgbImage::new(2, 2));
        let output = save_output(&input, &image).unwrap();

        assert_eq!(
            output,
            directory.canonicalize().unwrap().join("frame.pp.1.jpg")
        );
        assert_eq!(std::fs::read(&occupied).unwrap(), b"do-not-overwrite");
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn dedicated_output_directory_keeps_publication_out_of_the_input_tree() {
        let root = std::env::temp_dir().join(format!(
            "manwe-output-directory-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let _ = fs::remove_dir_all(&root);
        let input_dir = root.join("inputs");
        let output_dir = root.join("outputs");
        fs::create_dir_all(&input_dir).unwrap();
        fs::create_dir(&output_dir).unwrap();
        let input = input_dir.join("frame.png");
        fs::write(&input, b"input-marker").unwrap();

        let image = DynamicImage::ImageRgb8(image::RgbImage::new(2, 2));
        let output_directory = BoundDirectory::open(&output_dir).unwrap();
        let (output, digest) = save_output_with_digest(&input, &image, &output_directory).unwrap();

        assert_eq!(output.parent().unwrap(), output_dir.canonicalize().unwrap());
        assert!(!input_dir.join("frame.pp.jpg").exists());
        assert_eq!(sha256_hex(&fs::read(output).unwrap()), digest);
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn published_output_is_never_group_or_world_accessible() {
        use std::os::unix::fs::PermissionsExt;

        let directory = std::env::temp_dir().join(format!(
            "manwe-output-mode-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let input = directory.join("frame.png");
        fs::write(&input, b"input-marker").unwrap();

        let image = DynamicImage::ImageRgb8(image::RgbImage::new(2, 2));
        let output = save_output(&input, &image).unwrap();
        let mode = fs::metadata(&output).unwrap().permissions().mode();

        assert_eq!(mode & 0o077, 0);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn post_link_failure_reports_the_visible_path_and_preserves_evidence() {
        let directory = std::env::temp_dir().join(format!(
            "manwe-output-post-link-failure-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let mut publication = acquire_publication(&directory).unwrap();
        let output = publication.parent_dir.path().join("frame.pp.jpg");
        let stage_dir = publication.stage_dir.path().to_path_buf();

        let error = publication
            .publish_with_hooks(
                &encoded_test_jpeg(),
                OsStr::new("frame.pp"),
                |_path| anyhow::bail!("injected post-link authentication failure"),
                |_path| Ok(()),
            )
            .unwrap_err();

        let rendered = format!("{error:#}");
        assert!(rendered.contains(&output.display().to_string()));
        assert!(rendered.contains("content or visibility could not be authenticated"));
        assert!(output.is_file());
        assert!(stage_dir.join("output.jpg").is_file());
        drop(publication);
        assert!(output.is_file());
        assert!(stage_dir.join("output.jpg").is_file());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn pre_commit_sync_failure_reports_the_authenticated_path_and_unknown_durability() {
        let directory = std::env::temp_dir().join(format!(
            "manwe-output-pre-commit-sync-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let moved = directory.with_extension("moved");
        let _ = fs::remove_dir_all(&directory);
        let _ = fs::remove_dir_all(&moved);
        fs::create_dir(&directory).unwrap();
        let mut publication = acquire_publication(&directory).unwrap();
        let output = publication.parent_dir.path().join("frame.pp.jpg");

        let error = publication
            .publish_with_hooks(
                &encoded_test_jpeg(),
                OsStr::new("frame.pp"),
                |_path| Ok(()),
                |_path| {
                    fs::rename(&directory, &moved)?;
                    fs::create_dir(&directory)?;
                    Ok(())
                },
            )
            .unwrap_err();

        let rendered = format!("{error:#}");
        assert!(rendered.contains(&output.display().to_string()));
        assert!(rendered.contains("unknown publication durability"));
        assert!(moved.join("frame.pp.jpg").is_file());
        drop(publication);
        assert!(moved.join("frame.pp.jpg").is_file());
        fs::remove_dir_all(directory).unwrap();
        fs::remove_dir_all(moved).unwrap();
    }

    #[test]
    fn output_staging_failure_preserves_an_unexpected_directory_without_publishing() {
        let directory = std::env::temp_dir().join(format!(
            "manwe-output-failure-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let mut publication = acquire_publication(&directory).unwrap();
        let stage_dir = publication.stage_dir.path().to_path_buf();
        fs::create_dir(publication.stage_dir.path().join(STAGED_OUTPUT_NAME)).unwrap();
        let encoded = encoded_test_jpeg();

        assert!(publication
            .publish(&encoded, OsStr::new("frame.pp"))
            .is_err());
        drop(publication);

        assert!(!directory.join("frame.pp.jpg").exists());
        assert!(stage_dir.is_dir());
        assert!(stage_dir.join("output.jpg").is_dir());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn dropping_an_unused_publication_removes_only_its_empty_staging_entry() {
        let directory = std::env::temp_dir().join(format!(
            "manwe-output-empty-stage-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let publication = acquire_publication(&directory).unwrap();
        let stage_dir = publication.stage_dir.path().to_path_buf();

        drop(publication);

        assert!(!stage_dir.exists());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn output_cleanup_preserves_a_replacement_at_the_published_path() {
        let directory = std::env::temp_dir().join(format!(
            "manwe-output-replacement-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let mut publication = acquire_publication(&directory).unwrap();
        let encoded = encoded_test_jpeg();
        write_verified_jpeg_once(
            &publication.stage_dir,
            OsStr::new(STAGED_OUTPUT_NAME),
            &encoded,
        )
        .unwrap();
        let output = publication.parent_dir.path().join("frame.pp.jpg");
        fs::hard_link(
            publication.stage_dir.path().join(STAGED_OUTPUT_NAME),
            &output,
        )
        .unwrap();
        verify_jpeg_file_at(
            &publication.stage_dir,
            OsStr::new(STAGED_OUTPUT_NAME),
            &encoded,
        )
        .unwrap();
        publication.final_link_created = true;
        fs::remove_file(&output).unwrap();
        fs::write(&output, b"replacement-image").unwrap();
        let stage_dir = publication.stage_dir.path().to_path_buf();

        drop(publication);

        assert_eq!(fs::read(&output).unwrap(), b"replacement-image");
        assert!(stage_dir.is_dir());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn output_failure_preserves_its_exact_uncommitted_link_and_marker() {
        let directory = std::env::temp_dir().join(format!(
            "manwe-output-exact-cleanup-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let mut publication = acquire_publication(&directory).unwrap();
        let encoded = encoded_test_jpeg();
        write_verified_jpeg_once(
            &publication.stage_dir,
            OsStr::new(STAGED_OUTPUT_NAME),
            &encoded,
        )
        .unwrap();
        let output = publication.parent_dir.path().join("frame.pp.jpg");
        fs::hard_link(
            publication.stage_dir.path().join(STAGED_OUTPUT_NAME),
            &output,
        )
        .unwrap();
        verify_jpeg_file_at(
            &publication.stage_dir,
            OsStr::new(STAGED_OUTPUT_NAME),
            &encoded,
        )
        .unwrap();
        publication.final_link_created = true;
        let stage_dir = publication.stage_dir.path().to_path_buf();

        drop(publication);

        assert!(output.is_file());
        assert!(stage_dir.is_dir());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn output_publication_fails_closed_when_parent_directory_is_replaced() {
        let directory = std::env::temp_dir().join(format!(
            "manwe-output-directory-replacement-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let moved = directory.with_extension("moved");
        let _ = fs::remove_dir_all(&directory);
        let _ = fs::remove_dir_all(&moved);
        fs::create_dir(&directory).unwrap();
        let mut publication = acquire_publication(&directory).unwrap();
        fs::rename(&directory, &moved).unwrap();
        fs::create_dir(&directory).unwrap();
        let replacement = directory.join("frame.pp.jpg");
        fs::write(&replacement, b"replacement-directory").unwrap();
        let encoded = encoded_test_jpeg();

        assert!(publication
            .publish(&encoded, OsStr::new("frame.pp"))
            .is_err());
        drop(publication);

        assert_eq!(fs::read(&replacement).unwrap(), b"replacement-directory");
        fs::remove_dir_all(directory).unwrap();
        fs::remove_dir_all(moved).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn output_publication_rejects_a_group_writable_parent_before_staging() {
        use std::os::unix::fs::PermissionsExt;

        let directory = std::env::temp_dir().join(format!(
            "manwe-output-boundary-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o770)).unwrap();

        let result = acquire_publication(&directory);

        assert!(result.is_err());
        assert!(fs::read_dir(&directory).unwrap().next().is_none());
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700)).unwrap();
        fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn output_publication_rechecks_the_parent_boundary_before_linking() {
        use std::os::unix::fs::PermissionsExt;

        let directory = std::env::temp_dir().join(format!(
            "manwe-output-boundary-recheck-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let mut publication = acquire_publication(&directory).unwrap();
        let stage_dir = publication.stage_dir.path().to_path_buf();
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o770)).unwrap();

        let error = publication
            .publish(&encoded_test_jpeg(), OsStr::new("frame.pp"))
            .unwrap_err();

        assert!(format!("{error:#}").contains("group- or world-writable"));
        assert!(!directory.join("frame.pp.jpg").exists());
        drop(publication);
        assert!(stage_dir.is_dir());
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700)).unwrap();
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn output_cleanup_preserves_a_replacement_staging_directory() {
        let directory = std::env::temp_dir().join(format!(
            "manwe-output-stage-replacement-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let publication = acquire_publication(&directory).unwrap();
        let stage_dir = publication.stage_dir.path().to_path_buf();
        let detached_stage = directory.join("detached-original-stage");
        fs::rename(&stage_dir, &detached_stage).unwrap();
        fs::create_dir(&stage_dir).unwrap();
        let replacement_marker = stage_dir.join("replacement-marker");
        fs::write(&replacement_marker, b"do-not-delete").unwrap();

        drop(publication);

        assert_eq!(fs::read(&replacement_marker).unwrap(), b"do-not-delete");
        assert!(detached_stage.is_dir());
        fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn output_cleanup_never_follows_a_replacement_staging_symlink() {
        use std::os::unix::fs::symlink;

        let directory = std::env::temp_dir().join(format!(
            "manwe-output-stage-symlink-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let publication = acquire_publication(&directory).unwrap();
        let stage_dir = publication.stage_dir.path().to_path_buf();
        let detached_stage = directory.join("detached-original-stage");
        let victim = directory.join("victim");
        fs::create_dir(&victim).unwrap();
        let victim_marker = victim.join("victim-marker");
        fs::write(&victim_marker, b"do-not-delete").unwrap();
        fs::rename(&stage_dir, &detached_stage).unwrap();
        symlink(&victim, &stage_dir).unwrap();

        drop(publication);

        assert!(fs::symlink_metadata(&stage_dir)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(fs::read(&victim_marker).unwrap(), b"do-not-delete");
        assert!(detached_stage.is_dir());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn committed_output_reports_and_preserves_unexpected_staging_content() {
        let directory = std::env::temp_dir().join(format!(
            "manwe-output-committed-cleanup-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let mut publication = acquire_publication(&directory).unwrap();
        let stage_dir = publication.stage_dir.path().to_path_buf();
        let unexpected = stage_dir.join("unexpected-content");
        fs::write(&unexpected, b"do-not-delete").unwrap();

        let error = publication
            .publish(&encoded_test_jpeg(), OsStr::new("frame.pp"))
            .unwrap_err();

        assert!(format!("{error:#}").contains("output is committed"));
        assert!(directory.join("frame.pp.jpg").is_file());
        assert_eq!(fs::read(&unexpected).unwrap(), b"do-not-delete");
        drop(publication);
        assert_eq!(fs::read(&unexpected).unwrap(), b"do-not-delete");
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn cleanup_reports_a_parent_replacement_after_exact_entry_removal_without_retrying() {
        let directory = std::env::temp_dir().join(format!(
            "manwe-output-cleanup-sync-race-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let moved = directory.with_extension("moved");
        let _ = fs::remove_dir_all(&directory);
        let _ = fs::remove_dir_all(&moved);
        fs::create_dir(&directory).unwrap();
        let mut publication = acquire_publication(&directory).unwrap();
        let stage_name = publication.stage_name.clone();
        publication.cleanup_started = true;

        let result = publication.cleanup_staging_with_hook(false, || {
            fs::rename(&directory, &moved).unwrap();
            fs::create_dir(&directory).unwrap();
            fs::write(directory.join("replacement-marker"), b"do-not-delete").unwrap();
        });

        assert!(result.is_err());
        assert!(!moved.join(&stage_name).exists());
        assert!(!directory.join(&stage_name).exists());
        drop(publication);
        assert_eq!(
            fs::read(directory.join("replacement-marker")).unwrap(),
            b"do-not-delete"
        );
        fs::remove_dir_all(directory).unwrap();
        fs::remove_dir_all(moved).unwrap();
    }
}
