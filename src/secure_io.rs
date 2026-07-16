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

    /// Require this directory to be an owner-controlled mutation boundary.
    ///
    /// Cleanup operations trust the effective OS account (and root), but reject a
    /// directory that another account can mutate through mode bits or a mutating
    /// macOS extended ACL.
    pub fn require_owner_mutation_boundary(&self) -> Result<()> {
        self.verify()?;
        #[cfg(not(unix))]
        {
            anyhow::bail!("owner-controlled directory mutation requires Unix")
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;

            let metadata = self.file.metadata()?;
            // SAFETY: `geteuid` has no pointer arguments and only reads process credentials.
            let effective_uid = unsafe { libc::geteuid() };
            if metadata.uid() != effective_uid {
                anyhow::bail!(
                    "directory must be owned by the effective account: {}",
                    self.path.display()
                )
            }
            ensure_trusted_path_component(&self.path, &metadata, false)?;
            ensure_no_access_granting_extended_acl(&self.path)?;
            let mut ancestor = self.path.parent();
            while let Some(directory) = ancestor {
                let metadata = std::fs::symlink_metadata(directory).with_context(|| {
                    format!(
                        "failed to inspect mutation-boundary path component {}",
                        directory.display()
                    )
                })?;
                if metadata.file_type().is_symlink() || !metadata.is_dir() {
                    anyhow::bail!(
                        "mutation-boundary path component is not a canonical directory: {}",
                        directory.display()
                    )
                }
                ensure_trusted_path_component(directory, &metadata, true)?;
                ancestor = directory.parent();
            }
            self.verify()
        }
    }

    /// Remove one non-directory entry relative to this bound directory.
    pub fn remove_file_entry(&self, name: &std::ffi::OsStr) -> Result<()> {
        self.remove_entry(name, 0, false).map(|_| ())
    }

    /// Remove one non-directory entry if present, returning whether it existed.
    pub fn remove_file_entry_if_exists(&self, name: &std::ffi::OsStr) -> Result<bool> {
        self.remove_entry(name, 0, true)
    }

    /// Remove one empty directory entry relative to this bound directory.
    pub fn remove_directory_entry(&self, name: &std::ffi::OsStr) -> Result<()> {
        #[cfg(not(unix))]
        {
            let _ = name;
            anyhow::bail!("descriptor-relative entry removal requires Unix")
        }
        #[cfg(unix)]
        {
            self.remove_entry(name, libc::AT_REMOVEDIR, false)
                .map(|_| ())
        }
    }

    fn remove_entry(
        &self,
        name: &std::ffi::OsStr,
        flags: libc::c_int,
        missing_ok: bool,
    ) -> Result<bool> {
        #[cfg(not(unix))]
        {
            let _ = (name, flags, missing_ok);
            anyhow::bail!("descriptor-relative entry removal requires Unix")
        }
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            use std::os::unix::ffi::OsStrExt;

            let bytes = name.as_bytes();
            if bytes.is_empty() || bytes == b"." || bytes == b".." || bytes.contains(&b'/') {
                anyhow::bail!("directory entry name must be one non-special basename")
            }
            let encoded = std::ffi::CString::new(bytes)
                .context("directory entry name contains an interior NUL")?;
            self.verify()?;
            // SAFETY: `self.file` is a live directory descriptor and `encoded` is
            // one live NUL-terminated basename. `flags` is either 0 or AT_REMOVEDIR.
            if unsafe { libc::unlinkat(self.file.as_raw_fd(), encoded.as_ptr(), flags) } != 0 {
                let error = std::io::Error::last_os_error();
                if missing_ok && error.kind() == std::io::ErrorKind::NotFound {
                    return Ok(false);
                }
                return Err(error).with_context(|| {
                    format!(
                        "failed to remove bound directory entry {}",
                        self.path.join(name).display()
                    )
                });
            }
            self.verify()?;
            Ok(true)
        }
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

/// Inspect the current identity of an already-opened bounded regular file.
///
/// Unlike a pathname reopen, this remains bound to the caller's exact descriptor.
/// It is useful after metadata changes such as `fchmod`, which intentionally change
/// ctime and therefore require a fresh identity before authenticated reads.
pub fn bounded_open_file_identity(
    file: &File,
    path: &Path,
    max_bytes: u64,
) -> Result<FileIdentity> {
    if max_bytes == 0 {
        anyhow::bail!("file size limit must be positive")
    }
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to inspect opened file {}", path.display()))?;
    validate_metadata(path, &metadata, max_bytes)?;
    Ok(identity(&metadata))
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

/// A canonical executable path plus the authenticated descriptor opened at resolution time.
#[derive(Clone, Debug)]
pub struct ResolvedExecutable {
    path: std::path::PathBuf,
    file: std::sync::Arc<File>,
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

    /// Return the authenticated executable descriptor retained at resolution.
    #[must_use]
    pub fn file(&self) -> &File {
        &self.file
    }

    /// Require a structurally bounded native ELF or Mach-O executable container.
    ///
    /// This excludes interpreter scripts (including Linux `binfmt_misc` payloads
    /// that do not identify as a valid ELF container) from evidence-grade child
    /// execution. The trusted root/current-account boundary also covers kernel
    /// binary-format registration.
    pub fn require_native_executable(&self) -> Result<()> {
        validate_native_executable(&self.file, self.identity.len(), &self.path)
    }

    /// Recheck the retained descriptor, path identity, ownership, modes, and ACLs.
    ///
    /// The explicit trust boundary is root plus the process's effective OS account;
    /// an equally privileged process is outside this pathname-hardening model.
    pub fn verify(&self) -> Result<()> {
        self.verify_with_hook(|| {})
    }

    fn verify_with_hook(&self, after_trust_walk: impl FnOnce()) -> Result<()> {
        let retained = bounded_open_file_identity(&self.file, &self.path, MAX_EXECUTABLE_BYTES)?;
        if retained != self.identity {
            anyhow::bail!("resolved executable descriptor changed before spawn")
        }
        let (file, current) = open_bounded_regular_file(&self.path, MAX_EXECUTABLE_BYTES)
            .context("resolved executable identity changed before spawn")?;
        if current != self.identity {
            anyhow::bail!("resolved executable identity changed before spawn")
        }
        ensure_trusted_executable_location(&self.path, &file.metadata()?)
            .context("resolved executable location became untrusted before spawn")?;
        after_trust_walk();
        let (final_file, final_identity) =
            open_bounded_regular_file(&self.path, MAX_EXECUTABLE_BYTES)
                .context("resolved executable identity changed after the trust check")?;
        if final_identity != self.identity {
            anyhow::bail!("resolved executable identity changed after the trust check")
        }
        let final_retained =
            bounded_open_file_identity(&self.file, &self.path, MAX_EXECUTABLE_BYTES)?;
        if final_retained != self.identity {
            anyhow::bail!("resolved executable descriptor changed after the trust check")
        }
        drop(final_file);
        Ok(())
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, mut buffer: &mut [u8], mut offset: u64) -> Result<()> {
    use std::os::unix::fs::FileExt;

    while !buffer.is_empty() {
        let read = file.read_at(buffer, offset)?;
        if read == 0 {
            anyhow::bail!("native executable header is truncated")
        }
        offset = offset
            .checked_add(read as u64)
            .context("native executable header offset overflowed")?;
        buffer = &mut buffer[read..];
    }
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "android", test))]
fn decode_u16(bytes: [u8; 2], little_endian: bool) -> u16 {
    if little_endian {
        u16::from_le_bytes(bytes)
    } else {
        u16::from_be_bytes(bytes)
    }
}

fn decode_u32(bytes: [u8; 4], little_endian: bool) -> u32 {
    if little_endian {
        u32::from_le_bytes(bytes)
    } else {
        u32::from_be_bytes(bytes)
    }
}

#[cfg(target_os = "macos")]
fn decode_u64(bytes: [u8; 8], little_endian: bool) -> u64 {
    if little_endian {
        u64::from_le_bytes(bytes)
    } else {
        u64::from_be_bytes(bytes)
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn validate_native_executable(file: &File, file_len: u64, path: &Path) -> Result<()> {
    let mut header = [0_u8; 64];
    read_exact_at(file, &mut header, 0)
        .with_context(|| format!("failed to read ELF header from {}", path.display()))?;
    validate_elf_header(&header, file_len, path)
}

#[cfg(any(target_os = "linux", target_os = "android", test))]
fn validate_elf_header(header: &[u8; 64], file_len: u64, path: &Path) -> Result<()> {
    if &header[..4] != b"\x7fELF" {
        anyhow::bail!(
            "child executable is not a native ELF container: {}",
            path.display()
        )
    }
    let class = header[4];
    let little_endian = match header[5] {
        1 => true,
        2 => false,
        _ => anyhow::bail!(
            "ELF executable has an invalid byte order: {}",
            path.display()
        ),
    };
    if header[6] != 1 {
        anyhow::bail!(
            "ELF executable has an invalid identifier version: {}",
            path.display()
        )
    }
    let executable_type = decode_u16([header[16], header[17]], little_endian);
    let machine = decode_u16([header[18], header[19]], little_endian);
    let version = decode_u32(header[20..24].try_into()?, little_endian);
    let expected_header_size = match class {
        1 => 52_u16,
        2 => 64_u16,
        _ => anyhow::bail!("ELF executable has an invalid class: {}", path.display()),
    };
    let header_size_offset = if class == 1 { 40 } else { 52 };
    let header_size = decode_u16(
        header[header_size_offset..header_size_offset + 2].try_into()?,
        little_endian,
    );
    if !matches!(executable_type, 2 | 3)
        || machine == 0
        || version != 1
        || header_size != expected_header_size
        || file_len < u64::from(expected_header_size)
    {
        anyhow::bail!("ELF executable header is not loadable: {}", path.display())
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn validate_native_executable(file: &File, file_len: u64, path: &Path) -> Result<()> {
    let mut magic = [0_u8; 4];
    read_exact_at(file, &mut magic, 0)
        .with_context(|| format!("failed to read Mach-O magic from {}", path.display()))?;
    match magic {
        [0xce, 0xfa, 0xed, 0xfe] => validate_thin_macho(file, 0, file_len, true, false, path),
        [0xcf, 0xfa, 0xed, 0xfe] => validate_thin_macho(file, 0, file_len, true, true, path),
        [0xfe, 0xed, 0xfa, 0xce] => validate_thin_macho(file, 0, file_len, false, false, path),
        [0xfe, 0xed, 0xfa, 0xcf] => validate_thin_macho(file, 0, file_len, false, true, path),
        [0xca, 0xfe, 0xba, 0xbe] => validate_fat_macho(file, file_len, false, false, path),
        [0xbe, 0xba, 0xfe, 0xca] => validate_fat_macho(file, file_len, true, false, path),
        [0xca, 0xfe, 0xba, 0xbf] => validate_fat_macho(file, file_len, false, true, path),
        [0xbf, 0xba, 0xfe, 0xca] => validate_fat_macho(file, file_len, true, true, path),
        _ => anyhow::bail!(
            "child executable is not a native Mach-O container: {}",
            path.display()
        ),
    }
}

#[cfg(target_os = "macos")]
fn validate_thin_macho(
    file: &File,
    offset: u64,
    slice_len: u64,
    little_endian: bool,
    is_64_bit: bool,
    path: &Path,
) -> Result<()> {
    let header_len = if is_64_bit { 32_u64 } else { 28_u64 };
    if slice_len < header_len {
        anyhow::bail!("Mach-O executable header is truncated: {}", path.display())
    }
    let mut header = [0_u8; 32];
    read_exact_at(file, &mut header[..header_len as usize], offset)?;
    let file_type = decode_u32(header[12..16].try_into()?, little_endian);
    let command_count = decode_u32(header[16..20].try_into()?, little_endian);
    let command_bytes = decode_u32(header[20..24].try_into()?, little_endian);
    let command_end = header_len
        .checked_add(u64::from(command_bytes))
        .context("Mach-O load-command length overflowed")?;
    if file_type != 2 || command_count == 0 || command_bytes == 0 || command_end > slice_len {
        anyhow::bail!(
            "Mach-O executable header is not loadable: {}",
            path.display()
        )
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn validate_fat_macho(
    file: &File,
    file_len: u64,
    little_endian: bool,
    is_64_bit: bool,
    path: &Path,
) -> Result<()> {
    let mut count_bytes = [0_u8; 4];
    read_exact_at(file, &mut count_bytes, 4)?;
    let architecture_count = decode_u32(count_bytes, little_endian);
    if architecture_count == 0 || architecture_count > 64 {
        anyhow::bail!(
            "fat Mach-O architecture count is invalid: {}",
            path.display()
        )
    }
    let entry_len = if is_64_bit { 32_u64 } else { 20_u64 };
    let table_len = u64::from(architecture_count)
        .checked_mul(entry_len)
        .and_then(|length| length.checked_add(8))
        .context("fat Mach-O architecture table overflowed")?;
    if table_len > file_len {
        anyhow::bail!(
            "fat Mach-O architecture table is truncated: {}",
            path.display()
        )
    }
    let mut table = vec![0_u8; (table_len - 8) as usize];
    read_exact_at(file, &mut table, 8)?;
    for entry in table.chunks_exact(entry_len as usize) {
        let (offset, size) = if is_64_bit {
            (
                decode_u64(entry[8..16].try_into()?, little_endian),
                decode_u64(entry[16..24].try_into()?, little_endian),
            )
        } else {
            (
                u64::from(decode_u32(entry[8..12].try_into()?, little_endian)),
                u64::from(decode_u32(entry[12..16].try_into()?, little_endian)),
            )
        };
        let end = offset
            .checked_add(size)
            .context("fat Mach-O architecture range overflowed")?;
        if offset < table_len || size < 28 || end > file_len {
            anyhow::bail!(
                "fat Mach-O architecture range is invalid: {}",
                path.display()
            )
        }
        let mut magic = [0_u8; 4];
        read_exact_at(file, &mut magic, offset)?;
        match magic {
            [0xce, 0xfa, 0xed, 0xfe] => validate_thin_macho(file, offset, size, true, false, path)?,
            [0xcf, 0xfa, 0xed, 0xfe] => validate_thin_macho(file, offset, size, true, true, path)?,
            [0xfe, 0xed, 0xfa, 0xce] => {
                validate_thin_macho(file, offset, size, false, false, path)?
            }
            [0xfe, 0xed, 0xfa, 0xcf] => validate_thin_macho(file, offset, size, false, true, path)?,
            _ => anyhow::bail!("fat Mach-O contains a non-native slice: {}", path.display()),
        }
    }
    Ok(())
}

#[cfg(all(
    unix,
    not(any(target_os = "linux", target_os = "android", target_os = "macos"))
))]
fn validate_native_executable(_file: &File, _file_len: u64, path: &Path) -> Result<()> {
    anyhow::bail!(
        "native executable validation is unavailable on this Unix target: {}",
        path.display()
    )
}

#[cfg(not(unix))]
fn validate_native_executable(_file: &File, _file_len: u64, path: &Path) -> Result<()> {
    anyhow::bail!(
        "native executable validation requires Unix: {}",
        path.display()
    )
}

#[cfg(unix)]
fn ensure_trusted_executable_location(path: &Path, file_metadata: &Metadata) -> Result<()> {
    ensure_trusted_path_component(path, file_metadata, false)?;
    let mut ancestor = path.parent();
    while let Some(directory) = ancestor {
        let metadata = std::fs::symlink_metadata(directory).with_context(|| {
            format!(
                "failed to inspect executable path component {}",
                directory.display()
            )
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            anyhow::bail!(
                "executable path component is not a canonical directory: {}",
                directory.display()
            )
        }
        ensure_trusted_path_component(directory, &metadata, true)?;
        ancestor = directory.parent();
    }
    Ok(())
}

#[cfg(unix)]
fn ensure_trusted_path_component(
    component: &Path,
    metadata: &Metadata,
    allow_sticky_writable_directory: bool,
) -> Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    // The integrity boundary is the effective OS account: root and this account
    // are trusted, while other users and group/world-writable path components are
    // not. An equally privileged process can already alter this process directly.
    // SAFETY: `geteuid` has no pointer arguments and only reads process credentials.
    let effective_uid = unsafe { libc::geteuid() };
    let owner = metadata.uid();
    let mode = metadata.permissions().mode();
    if owner != 0 && owner != effective_uid {
        anyhow::bail!(
            "trusted path component is owned by an untrusted account: {}",
            component.display()
        )
    }
    let sticky_boundary = allow_sticky_writable_directory && mode & 0o1000 != 0;
    if mode & 0o022 != 0 && !sticky_boundary {
        anyhow::bail!(
            "trusted path component is group- or world-writable: {}",
            component.display()
        )
    }
    ensure_no_mutating_extended_acl(component)
}

#[cfg(target_os = "macos")]
fn ensure_no_mutating_extended_acl(path: &Path) -> Result<()> {
    const MUTATING_PERMISSIONS: u64 =
        (1 << 2) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 8) | (1 << 10) | (1 << 12) | (1 << 13);

    ensure_no_extended_acl_permissions(path, MUTATING_PERMISSIONS, "mutating")
}

#[cfg(target_os = "macos")]
fn ensure_no_access_granting_extended_acl(path: &Path) -> Result<()> {
    ensure_no_extended_acl_permissions(path, u64::MAX, "access-granting")
}

#[cfg(target_os = "macos")]
fn ensure_no_extended_acl_permissions(
    path: &Path,
    forbidden_permissions: u64,
    description: &str,
) -> Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    type Acl = *mut libc::c_void;
    type AclEntry = *mut libc::c_void;
    const ACL_TYPE_EXTENDED: libc::c_int = 0x0000_0100;
    const ACL_FIRST_ENTRY: libc::c_int = 0;
    const ACL_NEXT_ENTRY: libc::c_int = -1;
    const ACL_EXTENDED_ALLOW: libc::c_int = 1;

    unsafe extern "C" {
        fn acl_get_file(path: *const libc::c_char, acl_type: libc::c_int) -> Acl;
        fn acl_get_entry(acl: Acl, entry_id: libc::c_int, entry: *mut AclEntry) -> libc::c_int;
        fn acl_get_tag_type(entry: AclEntry, tag: *mut libc::c_int) -> libc::c_int;
        fn acl_get_permset_mask_np(entry: AclEntry, mask: *mut u64) -> libc::c_int;
        fn acl_free(object: *mut libc::c_void) -> libc::c_int;
    }

    let encoded = CString::new(path.as_os_str().as_bytes())
        .with_context(|| format!("path contains an interior NUL: {}", path.display()))?;
    // SAFETY: `encoded` is a live NUL-terminated path and the returned ACL is
    // released with `acl_free` before this function returns.
    let acl = unsafe { acl_get_file(encoded.as_ptr(), ACL_TYPE_EXTENDED) };
    if acl.is_null() {
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ENOENT) {
            return Ok(());
        }
        return Err(error).with_context(|| format!("failed to inspect ACL for {}", path.display()));
    }

    let inspection = (|| -> Result<()> {
        let mut entry: AclEntry = std::ptr::null_mut();
        let mut entry_id = ACL_FIRST_ENTRY;
        loop {
            // SAFETY: `acl` is live and `entry` points to writable storage for one
            // borrowed ACL-entry handle.
            let status = unsafe { acl_get_entry(acl, entry_id, &mut entry) };
            if status < 0 {
                let error = std::io::Error::last_os_error();
                if error.raw_os_error() == Some(libc::EINVAL) {
                    break;
                }
                return Err(error).with_context(|| {
                    format!("failed to inspect ACL entries for {}", path.display())
                });
            }
            let mut tag = 0;
            // SAFETY: `entry` was returned by `acl_get_entry`; `tag` is writable.
            if unsafe { acl_get_tag_type(entry, &mut tag) } != 0 {
                return Err(std::io::Error::last_os_error())
                    .with_context(|| format!("failed to inspect ACL tag for {}", path.display()));
            }
            if tag == ACL_EXTENDED_ALLOW {
                let mut permissions = 0_u64;
                // SAFETY: `entry` is live and `permissions` is writable storage for
                // the entry's permission mask.
                if unsafe { acl_get_permset_mask_np(entry, &mut permissions) } != 0 {
                    return Err(std::io::Error::last_os_error()).with_context(|| {
                        format!("failed to inspect ACL permissions for {}", path.display())
                    });
                }
                if permissions & forbidden_permissions != 0 {
                    anyhow::bail!(
                        "path component has a {description} extended ACL: {}",
                        path.display()
                    )
                }
            }
            entry_id = ACL_NEXT_ENTRY;
        }
        Ok(())
    })();
    // SAFETY: `acl` is the live allocation returned by `acl_get_file`.
    let free_status = unsafe { acl_free(acl) };
    if free_status != 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("failed to release ACL for {}", path.display()));
    }
    inspection
}

#[cfg(all(unix, not(target_os = "macos")))]
fn ensure_no_mutating_extended_acl(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(all(unix, not(target_os = "macos")))]
fn ensure_no_access_granting_extended_acl(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(not(unix))]
fn ensure_trusted_executable_location(_path: &Path, _file_metadata: &Metadata) -> Result<()> {
    anyhow::bail!("trusted executable path validation requires Unix")
}

/// Resolve an executable, authenticate it, retain its descriptor, and reject path
/// components mutable by accounts outside the explicit root/current-account trust
/// boundary.
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
    ensure_trusted_executable_location(&resolved, &metadata)?;
    let sha256 = sha256_bounded_open_file(
        &mut file,
        executable_identity,
        &resolved,
        MAX_EXECUTABLE_BYTES,
    )?;
    Ok(ResolvedExecutable {
        path: resolved,
        file: std::sync::Arc::new(file),
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
    fn owner_mutation_boundary_rejects_group_writable_directories() {
        use std::os::unix::fs::PermissionsExt;

        let directory = test_directory("group-writable-boundary");
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o770)).unwrap();
        let bound = BoundDirectory::open(&directory).unwrap();

        let error = bound.require_owner_mutation_boundary().unwrap_err();

        assert!(error.to_string().contains("group- or world-writable"));
        std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn owner_mutation_boundary_accepts_a_private_child_of_a_sticky_parent() {
        use std::os::unix::fs::PermissionsExt;

        let base = test_directory("sticky-parent-boundary");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir(&base).unwrap();
        let sticky_parent = base.join("shared");
        std::fs::create_dir(&sticky_parent).unwrap();
        std::fs::set_permissions(&sticky_parent, std::fs::Permissions::from_mode(0o1777)).unwrap();
        let directory = sticky_parent.join("private");
        std::fs::create_dir(&directory).unwrap();
        std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o700)).unwrap();
        let bound = BoundDirectory::open(&directory).unwrap();

        bound.require_owner_mutation_boundary().unwrap();

        std::fs::remove_dir_all(base).unwrap();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn owner_mutation_boundary_rejects_an_access_granting_acl() {
        let directory = test_directory("access-granting-boundary-acl");
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let status = std::process::Command::new("/bin/chmod")
            .args(["+a", "everyone allow read", directory.to_str().unwrap()])
            .status()
            .unwrap();
        assert!(status.success());
        let bound = BoundDirectory::open(&directory).unwrap();

        let error = bound.require_owner_mutation_boundary().unwrap_err();

        assert!(error.to_string().contains("access-granting extended ACL"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn descriptor_relative_removal_never_recurses_into_unknown_content() {
        let directory = test_directory("descriptor-relative-removal");
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let known = directory.join("known");
        std::fs::write(&known, b"known").unwrap();
        let unknown = directory.join("unknown");
        std::fs::create_dir(&unknown).unwrap();
        let sentinel = unknown.join("sentinel");
        std::fs::write(&sentinel, b"preserve").unwrap();
        let bound = BoundDirectory::open(&directory).unwrap();

        bound
            .remove_file_entry(std::ffi::OsStr::new("known"))
            .unwrap();
        let error = bound
            .remove_directory_entry(std::ffi::OsStr::new("unknown"))
            .unwrap_err();

        assert!(!known.exists());
        assert_eq!(std::fs::read(sentinel).unwrap(), b"preserve");
        assert!(error
            .to_string()
            .contains("failed to remove bound directory entry"));
        assert!(bound
            .remove_file_entry(std::ffi::OsStr::new("../escape"))
            .is_err());
        std::fs::remove_dir_all(directory).unwrap();
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

    #[cfg(unix)]
    #[test]
    fn opened_file_identity_refreshes_after_a_permission_change() {
        use std::os::unix::fs::PermissionsExt;

        let directory = test_directory("identity-after-mode-change");
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("artifact.bin");
        std::fs::write(&path, b"authenticated bytes").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o666)).unwrap();
        let (mut file, original) = open_bounded_regular_file(&path, 64).unwrap();

        file.set_permissions(std::fs::Permissions::from_mode(0o600))
            .unwrap();
        let refreshed = bounded_open_file_identity(&file, &path, 64).unwrap();
        let digest = sha256_bounded_open_file(&mut file, refreshed, &path, 64).unwrap();

        assert_ne!(original, refreshed);
        assert_eq!(digest, sha256_hex(b"authenticated bytes"));
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
        executable.require_native_executable().unwrap();
        assert!(executable.path().is_absolute());
        assert_eq!(executable.sha256().len(), 64);
    }

    #[cfg(unix)]
    #[test]
    fn executable_verification_rechecks_identity_after_the_trust_walk() {
        use std::os::unix::fs::PermissionsExt;

        let directory = test_directory("post-trust-executable-swap");
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let executable_path = directory.join("tool");
        std::fs::copy(std::env::current_exe().unwrap(), &executable_path).unwrap();
        std::fs::set_permissions(&executable_path, std::fs::Permissions::from_mode(0o700)).unwrap();
        let executable = resolve_executable(&executable_path).unwrap();

        let error = executable
            .verify_with_hook(|| {
                std::fs::remove_file(&executable_path).unwrap();
                std::fs::copy("/bin/echo", &executable_path).unwrap();
            })
            .unwrap_err();

        assert!(error.to_string().contains("after the trust check"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn resolved_executable_rejects_a_group_writable_path_component() {
        use std::os::unix::fs::PermissionsExt;

        let directory = test_directory("writable-executable-parent");
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let writable_parent = directory.join("writable");
        std::fs::create_dir(&writable_parent).unwrap();
        std::fs::set_permissions(&writable_parent, std::fs::Permissions::from_mode(0o770)).unwrap();
        let executable_path = writable_parent.join("tool");
        std::fs::copy(std::env::current_exe().unwrap(), &executable_path).unwrap();
        std::fs::set_permissions(&executable_path, std::fs::Permissions::from_mode(0o700)).unwrap();

        let error = resolve_executable(&executable_path).unwrap_err();

        assert!(error.to_string().contains("group- or world-writable"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn resolved_executable_rejects_a_mutating_extended_acl() {
        use std::os::unix::fs::PermissionsExt;

        let directory = test_directory("mutating-executable-acl");
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let executable_path = directory.join("tool");
        std::fs::copy(std::env::current_exe().unwrap(), &executable_path).unwrap();
        std::fs::set_permissions(&executable_path, std::fs::Permissions::from_mode(0o700)).unwrap();
        let status = std::process::Command::new("/bin/chmod")
            .args([
                "+a",
                "everyone allow write",
                executable_path.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(status.success());

        let error = resolve_executable(&executable_path).unwrap_err();

        assert!(error.to_string().contains("mutating extended ACL"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn evidence_grade_executable_boundary_rejects_interpreter_scripts() {
        use std::os::unix::fs::PermissionsExt;

        let directory = test_directory("interpreter-script");
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let script = directory.join("tool");
        std::fs::write(&script, b"#!/bin/sh\nexit 0\n").unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
        let executable = resolve_executable(&script).unwrap();

        let error = executable.require_native_executable().unwrap_err();

        assert!(error.to_string().contains("not a native"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_executable_validation_rejects_a_truncated_macho_header() {
        use std::os::unix::fs::PermissionsExt;

        let directory = test_directory("truncated-macho");
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        let executable_path = directory.join("tool");
        let mut bytes = vec![0_u8; 32];
        bytes[..4].copy_from_slice(&[0xcf, 0xfa, 0xed, 0xfe]);
        std::fs::write(&executable_path, bytes).unwrap();
        std::fs::set_permissions(&executable_path, std::fs::Permissions::from_mode(0o700)).unwrap();
        let executable = resolve_executable(&executable_path).unwrap();

        let error = executable.require_native_executable().unwrap_err();

        assert!(error.to_string().contains("not loadable"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn elf_header_validation_accepts_a_bounded_native_header_and_rejects_et_none() {
        let path = Path::new("synthetic-elf");
        let mut header = [0_u8; 64];
        header[..4].copy_from_slice(b"\x7fELF");
        header[4] = 2;
        header[5] = 1;
        header[6] = 1;
        header[16..18].copy_from_slice(&3_u16.to_le_bytes());
        header[18..20].copy_from_slice(&62_u16.to_le_bytes());
        header[20..24].copy_from_slice(&1_u32.to_le_bytes());
        header[52..54].copy_from_slice(&64_u16.to_le_bytes());

        validate_elf_header(&header, 64, path).unwrap();
        header[16..18].copy_from_slice(&0_u16.to_le_bytes());

        let error = validate_elf_header(&header, 64, path).unwrap_err();
        assert!(error.to_string().contains("not loadable"));
    }
}
