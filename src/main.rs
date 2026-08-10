use std::ffi::{OsStr, OsString};
use std::fs::{self, OpenOptions};
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use candle::{DType, Tensor};
use candle_nn::{Module, VarBuilder};
use clap::{Parser, ValueEnum};
use image::DynamicImage;
use manwe::model::{Multiples, YoloV8, YoloV8Pose};
use manwe::secure_io::{
    open_bounded_regular_file, read_bounded_open_file, read_bounded_regular_file, sha256_hex,
    BoundDirectory, FileIdentity, MAX_ENCODED_IMAGE_BYTES, MAX_MODEL_BYTES,
};
use manwe::{
    device, prepare_image, report_detect_with_output, report_pose_with_output, ImageTransform,
    ReportOutput,
};

const MAX_OUTPUT_ATTEMPTS: u32 = 10_000;
const MAX_OUTPUT_JPEG_BYTES: u64 = 256 * 1024 * 1024;
static OUTPUT_STAGE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, ValueEnum, Debug)]
enum Which {
    N,
    S,
    M,
    L,
    X,
}

#[derive(Clone, Copy, ValueEnum, Debug)]
enum YoloTask {
    Detect,
    Pose,
}

fn probability(value: &str) -> std::result::Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("{value:?} is not a number"))?;
    if parsed.is_finite() && (0.0..=1.0).contains(&parsed) {
        Ok(parsed)
    } else {
        Err("value must be finite and between 0 and 1".to_string())
    }
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

fn sha256_digest(value: &str) -> std::result::Result<String, String> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(value.to_ascii_lowercase())
    } else {
        Err("SHA-256 must contain exactly 64 hexadecimal characters".to_string())
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

    /// Local, non-symlinked model weights in safetensors format.
    #[arg(long, env = "MANWE_MODEL")]
    model: PathBuf,

    /// Expected SHA-256 for the exact model artifact.
    #[arg(long, env = "MANWE_MODEL_SHA256", value_parser = sha256_digest)]
    model_sha256: String,

    /// Which model variant to use.
    #[arg(long, value_enum, default_value_t = Which::S)]
    which: Which,

    /// One or more input images.
    #[arg(required = true, num_args = 1..)]
    images: Vec<PathBuf>,

    /// Threshold for the model confidence level.
    #[arg(long, default_value_t = 0.25, value_parser = probability)]
    confidence_threshold: f32,

    /// Threshold for non-maximum suppression.
    #[arg(long, default_value_t = 0.45, value_parser = probability)]
    nms_threshold: f32,

    /// The task to run.
    #[arg(long, value_enum, default_value_t = YoloTask::Detect)]
    task: YoloTask,

    /// Legend size; zero disables labels.
    #[arg(long, default_value_t = 14, value_parser = bounded_legend_size)]
    legend_size: u32,
}

impl Args {
    fn model_path(&self) -> Result<&Path> {
        if self.model.extension().and_then(|ext| ext.to_str()) != Some("safetensors") {
            anyhow::bail!(
                "model must be a .safetensors file: {}",
                self.model.display()
            )
        }
        Ok(&self.model)
    }
}

trait Task: Module + Sized {
    fn load(vb: VarBuilder, multiples: Multiples) -> candle::Result<Self>;
    fn report(
        pred: &Tensor,
        image: DynamicImage,
        transform: &ImageTransform,
        confidence_threshold: f32,
        nms_threshold: f32,
        legend_size: u32,
    ) -> candle::Result<DynamicImage>;
}

impl Task for YoloV8 {
    fn load(vb: VarBuilder, multiples: Multiples) -> candle::Result<Self> {
        YoloV8::load(vb, multiples, 80)
    }

    fn report(
        pred: &Tensor,
        image: DynamicImage,
        transform: &ImageTransform,
        confidence_threshold: f32,
        nms_threshold: f32,
        legend_size: u32,
    ) -> candle::Result<DynamicImage> {
        report_detect_with_output(
            pred,
            image,
            transform,
            confidence_threshold,
            nms_threshold,
            legend_size,
            ReportOutput::Stdout,
        )
    }
}

impl Task for YoloV8Pose {
    fn load(vb: VarBuilder, multiples: Multiples) -> candle::Result<Self> {
        YoloV8Pose::load(vb, multiples, 1, (17, 3))
    }

    fn report(
        pred: &Tensor,
        image: DynamicImage,
        transform: &ImageTransform,
        confidence_threshold: f32,
        nms_threshold: f32,
        legend_size: u32,
    ) -> candle::Result<DynamicImage> {
        report_pose_with_output(
            pred,
            image,
            transform,
            confidence_threshold,
            nms_threshold,
            legend_size,
            ReportOutput::Stdout,
        )
    }
}

fn create_private_directory(path: &Path) -> std::io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path)
}

fn verify_jpeg_file(path: &Path, expected: &[u8]) -> Result<FileIdentity> {
    if expected.is_empty() || expected.len() as u64 > MAX_OUTPUT_JPEG_BYTES {
        anyhow::bail!("annotated JPEG must contain between 1 and {MAX_OUTPUT_JPEG_BYTES} bytes")
    }
    let (mut file, identity) = open_bounded_regular_file(path, MAX_OUTPUT_JPEG_BYTES)?;
    let actual = read_bounded_open_file(&mut file, identity, path, MAX_OUTPUT_JPEG_BYTES)?;
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

fn write_verified_jpeg_once(path: &Path, encoded: &[u8]) -> Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        // The hard link published later exposes this exact inode. Set the mode
        // at creation so a permissive process umask cannot make it writable by
        // another account between linking and post-link authentication.
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(encoded)?;
    file.sync_all()?;
    drop(file);
    verify_jpeg_file(path, encoded)?;
    Ok(())
}

struct ImagePublication {
    parent_dir: BoundDirectory,
    stage_dir: BoundDirectory,
    stage_name: OsString,
    stage_file: PathBuf,
    final_link_created: bool,
    committed: bool,
    cleanup_started: bool,
}

impl ImagePublication {
    fn acquire(parent_dir: &Path) -> Result<Self> {
        let parent_dir = BoundDirectory::open(parent_dir)?;
        parent_dir.require_owner_mutation_boundary()?;
        for _ in 0..MAX_OUTPUT_ATTEMPTS {
            let sequence = OUTPUT_STAGE_COUNTER.fetch_add(1, Ordering::Relaxed);
            let stage_name = OsString::from(format!(
                ".manwe-image-output-{}-{sequence}.in-progress",
                std::process::id()
            ));
            let stage_path = parent_dir.path().join(&stage_name);
            parent_dir.verify()?;
            match create_private_directory(&stage_path) {
                Ok(()) => {
                    parent_dir.verify()?;
                    let stage_dir = match BoundDirectory::open(&stage_path) {
                        Ok(directory) => directory,
                        Err(error) => {
                            let cleanup_result = parent_dir
                                .remove_directory_entry(&stage_name)
                                .and_then(|()| parent_dir.sync());
                            if let Err(cleanup_error) = cleanup_result {
                                return Err(error.context(format!(
                                    "failed to bind the new staging directory; exact-entry cleanup was refused or its durability is unknown: {cleanup_error:#}"
                                )));
                            }
                            return Err(error);
                        }
                    };
                    let publication = Self {
                        stage_file: stage_dir.path().join("output.jpg"),
                        parent_dir,
                        stage_dir,
                        stage_name,
                        final_link_created: false,
                        committed: false,
                        cleanup_started: false,
                    };
                    publication.parent_dir.sync()?;
                    publication.stage_dir.require_owner_mutation_boundary()?;
                    return Ok(publication);
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
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
        write_verified_jpeg_once(&self.stage_file, encoded)?;
        self.stage_dir.sync()?;
        let mut output = None;
        for attempt in 0..MAX_OUTPUT_ATTEMPTS {
            let mut name = OsString::from(base);
            if attempt > 0 {
                name.push(format!(".{attempt}"));
            }
            name.push(".jpg");
            let candidate = self.parent_dir.path().join(name);
            self.parent_dir.verify()?;
            self.stage_dir.verify()?;
            match fs::hard_link(&self.stage_file, &candidate) {
                Ok(()) => {
                    self.final_link_created = true;
                    let mut authenticate = || -> Result<()> {
                        after_link(&candidate)?;
                        let stage_identity = verify_jpeg_file(&self.stage_file, encoded)?;
                        self.parent_dir.verify()?;
                        self.stage_dir.verify()?;
                        let output_identity = verify_jpeg_file(&candidate, encoded)?;
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
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
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

        let output_name = OsStr::new("output.jpg");
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

fn save_output(input: &Path, image: &DynamicImage) -> Result<PathBuf> {
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
    let parent_dir = input
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut publication = ImagePublication::acquire(parent_dir)?;
    publication.publish(encoded.get_ref(), &base)
}

fn decode_image(path: &Path) -> Result<DynamicImage> {
    let bytes = read_bounded_regular_file(path, MAX_ENCODED_IMAGE_BYTES)?;
    let mut reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .with_context(|| format!("failed to determine image format for {}", path.display()))?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(32_768);
    limits.max_image_height = Some(32_768);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    reader
        .decode()
        .with_context(|| format!("failed to decode image {}", path.display()))
}

fn run<T: Task>(args: Args) -> Result<()> {
    let device = device(args.cpu)?;
    let multiples = match args.which {
        Which::N => Multiples::n(),
        Which::S => Multiples::s(),
        Which::M => Multiples::m(),
        Which::L => Multiples::l(),
        Which::X => Multiples::x(),
    };
    let model_path = args.model_path()?;
    let model_bytes = read_bounded_regular_file(model_path, MAX_MODEL_BYTES)?;
    let actual_digest = sha256_hex(&model_bytes);
    if actual_digest != args.model_sha256 {
        anyhow::bail!("model SHA-256 does not match the expected digest")
    }
    let vb = VarBuilder::from_buffered_safetensors(model_bytes, DType::F32, &device)?;
    let model = T::load(vb, multiples)?;

    for image_path in &args.images {
        let original = decode_image(image_path)?;
        let (input, transform) = prepare_image(&original, 640, 32, &device)?;
        let predictions = model.forward(&input)?.squeeze(0)?;
        let annotated = T::report(
            &predictions,
            original,
            &transform,
            args.confidence_threshold,
            args.nms_threshold,
            args.legend_size,
        )?;
        let output = save_output(image_path, &annotated)?;
        println!("{}", output.display());
    }

    Ok(())
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _trace_guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    match args.task {
        YoloTask::Detect => run::<YoloV8>(args),
        YoloTask::Pose => run::<YoloV8Pose>(args),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoded_test_jpeg() -> Vec<u8> {
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(2, 2));
        let mut encoded = Cursor::new(Vec::new());
        image
            .write_to(&mut encoded, image::ImageFormat::Jpeg)
            .unwrap();
        encoded.into_inner()
    }

    #[test]
    fn probability_rejects_nonfinite_and_out_of_range_values() {
        assert!(probability("NaN").is_err());
        assert!(probability("-0.1").is_err());
        assert!(probability("1.1").is_err());
        assert_eq!(probability("0.5").unwrap(), 0.5);
        assert!(sha256_digest(&"0".repeat(63)).is_err());
        assert_eq!(sha256_digest(&"A".repeat(64)).unwrap(), "a".repeat(64));
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
        let mut publication = ImagePublication::acquire(&directory).unwrap();
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
        let mut publication = ImagePublication::acquire(&directory).unwrap();
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
        let mut publication = ImagePublication::acquire(&directory).unwrap();
        let stage_dir = publication.stage_dir.path().to_path_buf();
        fs::create_dir(&publication.stage_file).unwrap();
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
        let publication = ImagePublication::acquire(&directory).unwrap();
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
        let mut publication = ImagePublication::acquire(&directory).unwrap();
        let encoded = encoded_test_jpeg();
        write_verified_jpeg_once(&publication.stage_file, &encoded).unwrap();
        let output = publication.parent_dir.path().join("frame.pp.jpg");
        fs::hard_link(&publication.stage_file, &output).unwrap();
        verify_jpeg_file(&publication.stage_file, &encoded).unwrap();
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
        let mut publication = ImagePublication::acquire(&directory).unwrap();
        let encoded = encoded_test_jpeg();
        write_verified_jpeg_once(&publication.stage_file, &encoded).unwrap();
        let output = publication.parent_dir.path().join("frame.pp.jpg");
        fs::hard_link(&publication.stage_file, &output).unwrap();
        verify_jpeg_file(&publication.stage_file, &encoded).unwrap();
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
        let mut publication = ImagePublication::acquire(&directory).unwrap();
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

        let result = ImagePublication::acquire(&directory);

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
        let mut publication = ImagePublication::acquire(&directory).unwrap();
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
        let publication = ImagePublication::acquire(&directory).unwrap();
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
        let publication = ImagePublication::acquire(&directory).unwrap();
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
        let mut publication = ImagePublication::acquire(&directory).unwrap();
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
        let mut publication = ImagePublication::acquire(&directory).unwrap();
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
