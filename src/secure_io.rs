//! Bounded, race-aware reads for local artifacts and benchmark inputs.

#[cfg(unix)]
use std::fs::OpenOptions;
use std::fs::{File, Metadata};
use std::io::{Read, Seek, Take};
use std::path::Path;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

/// Maximum accepted model size for the Rust inference surfaces (1 GiB).
pub const MAX_MODEL_BYTES: u64 = 1024 * 1024 * 1024;
/// Maximum accepted encoded image size for benchmark inputs (64 MiB).
pub const MAX_ENCODED_IMAGE_BYTES: u64 = 64 * 1024 * 1024;
/// Maximum accepted local video size for a benchmark run (8 GiB).
pub const MAX_VIDEO_BYTES: u64 = 8 * 1024 * 1024 * 1024;
/// Maximum accepted executable size for a verified child-process boundary (1 GiB).
pub const MAX_EXECUTABLE_BYTES: u64 = 1024 * 1024 * 1024;

/// Lowercase hexadecimal alphabet for every digest this workspace renders.
const HEX_LOWER: [u8; 16] = *b"0123456789abcdef";

/// Render bytes as lowercase, zero-padded hexadecimal.
///
/// `digest` 0.11 moved fixed-size outputs from `generic-array` to `hybrid-array`,
/// which deliberately does not implement `LowerHex`. This reproduces exactly what
/// the previous `format!("{:x}", ..)` emitted: two lowercase digits per byte, most
/// significant nibble first, zero-padded, never truncated. Digests are compared for
/// equality against operator-supplied values, so the rendering is load-bearing and
/// is pinned by known-answer tests below.
fn hex_lower(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len().saturating_mul(2));
    for &byte in bytes {
        encoded.push(char::from(HEX_LOWER[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX_LOWER[usize::from(byte & 0x0f)]));
    }
    encoded
}

/// Hash an in-memory buffer and render the digest as lowercase hexadecimal.
///
/// Every SHA-256 the workspace compares against an expected digest must be rendered
/// identically, so callers use this rather than formatting a `sha2` output directly.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

/// Opaque identity of one opened regular file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FileIdentity {
    len: u64,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    modified_seconds: i64,
    #[cfg(unix)]
    modified_nanoseconds: i64,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
    #[cfg(not(unix))]
    modified: Option<std::time::SystemTime>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DirectoryIdentity {
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(not(unix))]
    unsupported: (),
}

/// An opened directory bound to the device/inode identity of its canonical path.
#[derive(Debug)]
pub struct BoundDirectory {
    path: std::path::PathBuf,
    file: File,
    identity: DirectoryIdentity,
}

impl FileIdentity {
    /// Return the byte length observed through the opened file descriptor.
    #[must_use]
    pub fn len(self) -> u64 {
        self.len
    }

    /// Return whether the identity describes an empty file.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.len == 0
    }
}

#[cfg(unix)]
fn identity(metadata: &Metadata) -> FileIdentity {
    use std::os::unix::fs::MetadataExt;

    FileIdentity {
        len: metadata.len(),
        device: metadata.dev(),
        inode: metadata.ino(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
    }
}

#[cfg(unix)]
fn directory_identity(metadata: &Metadata) -> DirectoryIdentity {
    use std::os::unix::fs::MetadataExt;

    DirectoryIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    }
}

#[cfg(not(unix))]
fn directory_identity(_metadata: &Metadata) -> DirectoryIdentity {
    DirectoryIdentity { unsupported: () }
}

impl BoundDirectory {
    /// Open a directory without following its final symlink component.
    ///
    /// The canonical path and opened descriptor must identify the same directory.
    /// Non-Unix platforms fail closed until equivalent reparse-point handling exists.
    pub fn open(path: &Path) -> Result<Self> {
        #[cfg(not(unix))]
        {
            let _ = path;
            anyhow::bail!("race-resistant directory binding is not implemented on this platform")
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;

            let original_path = path;
            let path_metadata = std::fs::symlink_metadata(original_path)
                .with_context(|| format!("failed to inspect directory {}", path.display()))?;
            if path_metadata.file_type().is_symlink() || !path_metadata.is_dir() {
                anyhow::bail!("path must identify a directory: {}", path.display())
            }
            let mut options = OpenOptions::new();
            options
                .read(true)
                .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_DIRECTORY);
            let file = options.open(original_path).with_context(|| {
                format!(
                    "failed to bind directory without following links: {}",
                    original_path.display()
                )
            })?;
            let metadata = file.metadata()?;
            if !metadata.is_dir() {
                anyhow::bail!(
                    "opened path is not a directory: {}",
                    original_path.display()
                )
            }
            let path = original_path.canonicalize().with_context(|| {
                format!("failed to resolve directory {}", original_path.display())
            })?;
            let canonical_metadata = std::fs::symlink_metadata(&path)
                .with_context(|| format!("failed to inspect directory {}", path.display()))?;
            if canonical_metadata.file_type().is_symlink()
                || !canonical_metadata.is_dir()
                || directory_identity(&canonical_metadata) != directory_identity(&metadata)
            {
                anyhow::bail!(
                    "opened directory identity does not match its canonical path: {}",
                    path.display()
                )
            }
            let bound = Self {
                path,
                identity: directory_identity(&metadata),
                file,
            };
            bound.verify()?;
            Ok(bound)
        }
    }

    /// Return the canonical path whose identity is bound by this descriptor.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Fail if the canonical path no longer identifies the opened directory.
    pub fn verify(&self) -> Result<()> {
        #[cfg(not(unix))]
        {
            anyhow::bail!(
                "race-resistant directory verification is not implemented on this platform"
            )
        }
        #[cfg(unix)]
        {
            let path_metadata = std::fs::symlink_metadata(&self.path)
                .with_context(|| format!("failed to recheck directory {}", self.path.display()))?;
            if path_metadata.file_type().is_symlink()
                || !path_metadata.is_dir()
                || directory_identity(&path_metadata) != self.identity
                || directory_identity(&self.file.metadata()?) != self.identity
            {
                anyhow::bail!(
                    "directory identity changed during the operation: {}",
                    self.path.display()
                )
            }
            Ok(())
        }
    }

    /// Synchronize the opened directory after verifying its path binding.
    pub fn sync(&self) -> Result<()> {
        self.verify()?;
        self.file.sync_all()?;
        self.verify()
    }
}

#[cfg(not(unix))]
fn identity(metadata: &Metadata) -> FileIdentity {
    FileIdentity {
        len: metadata.len(),
        modified: metadata.modified().ok(),
    }
}

fn validate_metadata(path: &Path, metadata: &Metadata, max_bytes: u64) -> Result<()> {
    if !metadata.is_file() {
        anyhow::bail!("input must be a regular file: {}", path.display())
    }
    if metadata.len() == 0 {
        anyhow::bail!("input must not be empty: {}", path.display())
    }
    if metadata.len() > max_bytes {
        anyhow::bail!(
            "input exceeds the {max_bytes}-byte limit: {}",
            path.display()
        )
    }
    Ok(())
}

/// Open a nonempty, bounded regular file without following its final symlink component.
///
/// On Unix, `O_NOFOLLOW` closes the check/open race for the final component. The
/// returned identity describes the opened file descriptor, not a later path lookup.
/// Other platforms fail closed until an equivalent reparse-point-safe open exists.
pub fn open_bounded_regular_file(path: &Path, max_bytes: u64) -> Result<(File, FileIdentity)> {
    if max_bytes == 0 {
        anyhow::bail!("file size limit must be positive")
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        anyhow::bail!(
            "race-resistant regular-file opens are not implemented on this platform; refusing to follow reparse points"
        )
    }
    #[cfg(unix)]
    {
        let path_metadata = std::fs::symlink_metadata(path)
            .with_context(|| format!("failed to inspect {}", path.display()))?;
        if path_metadata.file_type().is_symlink() {
            anyhow::bail!("input must not be a symbolic link: {}", path.display())
        }

        let mut options = OpenOptions::new();
        options.read(true);
        use std::os::unix::fs::OpenOptionsExt;

        options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
        let file = options.open(path).with_context(|| {
            format!("failed to open {} without following links", path.display())
        })?;
        let metadata = file
            .metadata()
            .with_context(|| format!("failed to inspect opened file {}", path.display()))?;
        validate_metadata(path, &metadata, max_bytes)?;
        Ok((file, identity(&metadata)))
    }
}

fn bounded_reader(file: &mut File, max_bytes: u64) -> Take<&mut File> {
    file.take(max_bytes.saturating_add(1))
}

/// Read a nonempty regular file without following its final symlink component.
///
/// The read is capped independently of the initial metadata and fails if the
/// opened file changes while bytes are being consumed.
pub fn read_bounded_regular_file(path: &Path, max_bytes: u64) -> Result<Vec<u8>> {
    read_bounded_regular_file_with_identity(path, max_bytes).map(|(bytes, _)| bytes)
}

/// Read a bounded regular file and return the identity of the opened file.
pub fn read_bounded_regular_file_with_identity(
    path: &Path,
    max_bytes: u64,
) -> Result<(Vec<u8>, FileIdentity)> {
    let (mut file, before) = open_bounded_regular_file(path, max_bytes)?;
    let bytes = read_bounded_open_file(&mut file, before, path, max_bytes)?;
    Ok((bytes, before))
}

/// Read an already-opened bounded file whose identity was previously inspected.
pub fn read_bounded_open_file(
    file: &mut File,
    expected: FileIdentity,
    path: &Path,
    max_bytes: u64,
) -> Result<Vec<u8>> {
    if max_bytes == 0 {
        anyhow::bail!("file size limit must be positive")
    }
    let before_metadata = file
        .metadata()
        .with_context(|| format!("failed to inspect opened file {}", path.display()))?;
    validate_metadata(path, &before_metadata, max_bytes)?;
    if identity(&before_metadata) != expected {
        anyhow::bail!("input identity changed before reading: {}", path.display())
    }
    file.rewind()
        .with_context(|| format!("failed to rewind {}", path.display()))?;
    let capacity = usize::try_from(expected.len())
        .context("input size cannot be represented on this platform")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| anyhow::anyhow!("failed to reserve memory for bounded input"))?;
    bounded_reader(file, max_bytes)
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read {}", path.display()))?;
    if bytes.len() as u64 > max_bytes {
        anyhow::bail!(
            "input grew beyond the {max_bytes}-byte limit while being read: {}",
            path.display()
        )
    }
    let after = identity(&file.metadata()?);
    if expected != after || bytes.len() as u64 != expected.len() {
        anyhow::bail!("input changed while being read: {}", path.display())
    }
    file.rewind()
        .with_context(|| format!("failed to rewind {}", path.display()))?;
    Ok(bytes)
}

/// Hash a bounded regular file and return the identity of the exact opened file.
pub fn sha256_bounded_regular_file(path: &Path, max_bytes: u64) -> Result<(String, FileIdentity)> {
    let (mut file, before) = open_bounded_regular_file(path, max_bytes)?;
    let digest = sha256_bounded_open_file(&mut file, before, path, max_bytes)?;
    Ok((digest, before))
}

/// Hash an already-opened bounded file, then rewind it for a subsequent consumer.
///
/// This is useful when a child process must consume the exact file descriptor
/// whose bytes were authenticated. The identity is checked both before and after
/// hashing, and the byte count must still match the original metadata.
pub fn sha256_bounded_open_file(
    file: &mut File,
    expected: FileIdentity,
    path: &Path,
    max_bytes: u64,
) -> Result<String> {
    if max_bytes == 0 {
        anyhow::bail!("file size limit must be positive")
    }
    let before_metadata = file
        .metadata()
        .with_context(|| format!("failed to inspect opened file {}", path.display()))?;
    validate_metadata(path, &before_metadata, max_bytes)?;
    if identity(&before_metadata) != expected {
        anyhow::bail!("input identity changed before hashing: {}", path.display())
    }
    file.rewind()
        .with_context(|| format!("failed to rewind {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    {
        let mut reader = bounded_reader(file, max_bytes);
        loop {
            let read = reader
                .read(&mut buffer)
                .with_context(|| format!("failed to hash {}", path.display()))?;
            if read == 0 {
                break;
            }
            total = total
                .checked_add(read as u64)
                .context("input byte count overflowed")?;
            if total > max_bytes {
                anyhow::bail!(
                    "input grew beyond the {max_bytes}-byte limit while being hashed: {}",
                    path.display()
                )
            }
            hasher.update(&buffer[..read]);
        }
    }
    let after = identity(&file.metadata()?);
    if expected != after || total != expected.len() {
        anyhow::bail!("input changed while being hashed: {}", path.display())
    }
    file.rewind()
        .with_context(|| format!("failed to rewind {}", path.display()))?;
    Ok(hex_lower(&hasher.finalize()))
}

/// Fail if a path no longer identifies the previously opened regular file.
pub fn ensure_file_identity(path: &Path, expected: FileIdentity, max_bytes: u64) -> Result<()> {
    let (file, current) = open_bounded_regular_file(path, max_bytes)?;
    drop(file);
    if current != expected {
        anyhow::bail!("input identity changed during the run: {}", path.display())
    }
    Ok(())
}

/// A canonical executable path bound to the regular-file identity inspected at resolution time.
#[derive(Clone, Debug)]
pub struct ResolvedExecutable {
    path: std::path::PathBuf,
    identity: FileIdentity,
    sha256: String,
}

impl ResolvedExecutable {
    /// Return the canonical executable path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Return the SHA-256 of the executable bytes inspected at resolution time.
    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Recheck that the path still names the executable resolved earlier.
    ///
    /// This narrows pathname replacement races before `Command::spawn`. Callers
    /// should still place executables in directories not writable by an attacker.
    pub fn verify(&self) -> Result<()> {
        ensure_file_identity(&self.path, self.identity, MAX_EXECUTABLE_BYTES)
            .context("resolved executable identity changed before spawn")
    }
}

/// Resolve an executable and retain its regular-file identity for pre-spawn checks.
pub fn resolve_executable(path: &Path) -> Result<ResolvedExecutable> {
    let candidate = if path.is_absolute() || path.components().count() > 1 {
        path.to_path_buf()
    } else {
        let search_path = std::env::var_os("PATH").context("PATH is unavailable")?;
        std::env::split_paths(&search_path)
            .map(|directory| directory.join(path))
            .find(|candidate| candidate.is_file())
            .context("executable was not found in PATH")?
    };
    let resolved = candidate
        .canonicalize()
        .with_context(|| format!("failed to resolve executable {}", candidate.display()))?;
    let (mut file, executable_identity) =
        open_bounded_regular_file(&resolved, MAX_EXECUTABLE_BYTES)?;
    let metadata = file.metadata()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        if metadata.permissions().mode() & 0o111 == 0 {
            anyhow::bail!("resolved path is not executable")
        }
    }
    let sha256 = sha256_bounded_open_file(
        &mut file,
        executable_identity,
        &resolved,
        MAX_EXECUTABLE_BYTES,
    )?;
    Ok(ResolvedExecutable {
        path: resolved,
        identity: executable_identity,
        sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_directory(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "manwe-secure-io-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    #[test]
    fn bounded_read_rejects_oversized_files() {
        let directory = test_directory("oversized");
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("artifact.bin");
        std::fs::write(&path, b"12345").unwrap();

        let error = read_bounded_regular_file(&path, 4).unwrap_err();

        assert!(error.to_string().contains("4-byte limit"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn identity_detects_path_replacement() {
        let directory = test_directory("replacement");
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("artifact.bin");
        std::fs::write(&path, b"first").unwrap();
        let (_, identity) = sha256_bounded_regular_file(&path, 16).unwrap();
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, b"other").unwrap();

        let error = ensure_file_identity(&path, identity, 16).unwrap_err();

        assert!(error.to_string().contains("identity changed"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn bound_directory_detects_path_replacement() {
        let directory = test_directory("directory-replacement");
        let moved = directory.with_extension("moved");
        let _ = std::fs::remove_dir_all(&directory);
        let _ = std::fs::remove_dir_all(&moved);
        std::fs::create_dir(&directory).unwrap();
        let bound = BoundDirectory::open(&directory).unwrap();
        std::fs::rename(&directory, &moved).unwrap();
        std::fs::create_dir(&directory).unwrap();

        let error = bound.verify().unwrap_err();

        assert!(error.to_string().contains("directory identity changed"));
        drop(bound);
        std::fs::remove_dir_all(directory).unwrap();
        std::fs::remove_dir_all(moved).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn bound_directory_rejects_a_final_symlink() {
        use std::os::unix::fs::symlink;

        let directory = test_directory("directory-symlink");
        let link = directory.with_extension("link");
        let _ = std::fs::remove_dir_all(&directory);
        let _ = std::fs::remove_file(&link);
        std::fs::create_dir(&directory).unwrap();
        symlink(&directory, &link).unwrap();

        let error = BoundDirectory::open(&link).unwrap_err();

        assert!(error.to_string().contains("must identify a directory"));
        std::fs::remove_file(link).unwrap();
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn opened_file_hashing_rewinds_the_authenticated_descriptor() {
        let directory = test_directory("opened-hash");
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("artifact.bin");
        std::fs::write(&path, b"authenticated bytes").unwrap();
        let (mut file, identity) = open_bounded_regular_file(&path, 64).unwrap();

        let first = sha256_bounded_open_file(&mut file, identity, &path, 64).unwrap();
        let bytes = read_bounded_open_file(&mut file, identity, &path, 64).unwrap();
        let second = sha256_bounded_open_file(&mut file, identity, &path, 64).unwrap();

        assert_eq!(first, second);
        assert_eq!(bytes, b"authenticated bytes");
        assert_eq!(first, sha256_hex(b"authenticated bytes"));
        assert_eq!(
            first,
            "4a79516c84b8144eb7ba196298962abd826363a6481cb4a9dacba815610dacf7"
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    /// Pin the digest rendering against vectors computed by an independent
    /// implementation. A silently wrong encoder here would break every integrity
    /// check in the workspace, so these must never be relaxed to self-comparison.
    #[test]
    fn sha256_hex_matches_known_answer_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // A digest whose leading byte is 0x00: catches an encoder that drops
        // zero padding and would otherwise emit 63 characters.
        let leading_zero = sha256_hex(b"manwe-131");
        assert_eq!(
            leading_zero,
            "0098e64429590bbae80de3cd7430d4b8460bdf1f1335546e519877adb293d784"
        );
        assert_eq!(leading_zero.len(), 64);
        assert!(leading_zero
            .chars()
            .all(|character| character.is_ascii_digit() || ('a'..='f').contains(&character)));
    }

    /// The streaming path must agree with the one-shot digest across the internal
    /// 64 KiB buffer boundary, so a partial `update` loop cannot go unnoticed.
    #[test]
    fn streamed_hashing_matches_the_one_shot_digest_across_buffer_boundaries() {
        let directory = test_directory("streaming-hash");
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();

        for length in [1_usize, 65_535, 65_536, 65_537, 200_000] {
            let path = directory.join(format!("artifact-{length}.bin"));
            let contents: Vec<u8> = (0..length).map(|index| (index % 251) as u8).collect();
            std::fs::write(&path, &contents).unwrap();

            let (streamed, identity) = sha256_bounded_regular_file(&path, 1 << 20).unwrap();

            assert_eq!(streamed, sha256_hex(&contents));
            assert_eq!(identity.len(), length as u64);
            std::fs::remove_file(&path).unwrap();
        }
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn bounded_read_rejects_symbolic_links() {
        use std::os::unix::fs::symlink;

        let directory = test_directory("symlink");
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let target = directory.join("target.bin");
        let link = directory.join("link.bin");
        std::fs::write(&target, b"bytes").unwrap();
        symlink(&target, &link).unwrap();

        let error = read_bounded_regular_file(&link, 16).unwrap_err();

        assert!(error.to_string().contains("symbolic link"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn resolved_executable_retains_a_recheckable_identity() {
        let executable = resolve_executable(&std::env::current_exe().unwrap()).unwrap();

        executable.verify().unwrap();
        assert!(executable.path().is_absolute());
        assert_eq!(executable.sha256().len(), 64);
    }
}
