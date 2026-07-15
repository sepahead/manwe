"""Conservative, non-echoing credential scan used by local checks and CI.

The default mode scans tracked and non-ignored untracked worktree files.  A
separate revision-range mode scans every new Git blob reachable from commits in
``BASE..HEAD``.  The latter catches credentials that were introduced and then
deleted before the tip of a pull request.

Only rule names and locations are returned; matched values are never retained
in findings or written to diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .config_io import read_bounded_regular_bytes

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_GIT_OBJECTS = 100_000
MAX_GIT_SCAN_BYTES = 256 * 1024 * 1024
_MAX_GIT_OBJECT_ID_LINE_BYTES = 66


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str


TOKEN_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private-key", re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")),
    (
        "github-token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    ),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("stripe-live-key", re.compile(r"\b[rs]k_live_[A-Za-z0-9]{16,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("openai-token", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("huggingface-token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("gitlab-token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b")),
    ("npm-token", re.compile(r"\bnpm_[A-Za-z0-9]{20,}\b")),
    ("pypi-token", re.compile(r"\bpypi-[A-Za-z0-9_-]{30,}\b")),
)

_SECRET_NAME = (
    r"api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
    r"password|passwd|private[_-]?key|secret|"
    r"(?:aws|openai|hf|huggingface|gitlab|npm|pypi|github|slack)[_-]?token"
)
GENERIC_ASSIGNMENT = re.compile(
    rf"""(?ix)
    (?<![\w-])(?P<key_quote>[\"']?)
    (?:[A-Z0-9]+[_-])*(?:{_SECRET_NAME})(?:[_-][A-Z0-9]+)*
    (?P=key_quote)
    \s*[:=]\s*
    (?:
        \"(?P<double_quoted_value>(?:\\[^\r\n]|[^\"\\\r\n])+)\"
        |
        '(?P<single_quoted_value>(?:\\[^\r\n]|[^'\\\r\n])+)'
        |
        (?P<unquoted_value>[^\s,;#]+)
    )
    """
)

AUTHORIZATION_ASSIGNMENT = re.compile(
    r"""(?ix)
    (?<![\w-])(?P<key_quote>[\"']?)authorization(?P=key_quote)
    \s*[:=]\s*
    (?:
        \"(?P<double_quoted_value>(?:\\[^\r\n]|[^\"\\\r\n])+)\"
        |
        '(?P<single_quoted_value>(?:\\[^\r\n]|[^'\\\r\n])+)'
        |
        (?P<unquoted_value>(?:(?:basic|bearer|token)[ \t]+)?[^\s\"',;#]+)
    )
    """
)

BEARER_TOKEN = re.compile(
    r"""(?ix)
    \bbearer[ \t]+
    (?:
        \"(?P<double_quoted_value>(?:\\[^\r\n]|[^\"\\\r\n])+)\"
        |
        '(?P<single_quoted_value>(?:\\[^\r\n]|[^'\\\r\n])+)'
        |
        (?P<unquoted_value>[A-Za-z0-9_$<{][^\s\"',;#]*)
    )
    """
)

CREDENTIALED_URL = re.compile(
    r"""(?ix)
    \b(?:https?|rtsps?|postgres(?:ql)?|mysql|mongodb(?:\+srv)?|rediss?|amqps?)
    ://
    (?P<user>[^:/@\s\"'?#]+):(?P<password>[^/@\s\"'?#]+)@
    (?P<host>\[[^\]]+\]|[^/:\s\"'?#]+)
    """
)

# These exemptions match the complete credential value.  In particular,
# ``${TOKEN}-suffix`` and ``example-value-real`` are not placeholders.
PLACEHOLDER_VALUES = frozenset(
    {
        "change-me",
        "changeme",
        "dummy-value",
        "example-value",
        "not-a-real-secret",
        "pass",
        "password",
        "placeholder",
        "redacted",
        "replace-me",
        "test-only",
        "user",
        "username",
    }
)
PLACEHOLDER_EXPRESSIONS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}"),
    re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*"),
    re.compile(r"<[A-Za-z_][A-Za-z0-9_.-]*>"),
    re.compile(r"\{\{\s*[A-Za-z_][A-Za-z0-9_.-]*\s*\}\}"),
    re.compile(r"\$\{\{\s*secrets\.[A-Za-z_][A-Za-z0-9_]*\s*\}\}"),
)


def _is_placeholder(value: str) -> bool:
    normalized = value.strip()
    return normalized.lower() in PLACEHOLDER_VALUES or any(
        pattern.fullmatch(normalized) for pattern in PLACEHOLDER_EXPRESSIONS
    )


def _match_value(match: re.Match[str]) -> str:
    return (
        match.group("double_quoted_value")
        or match.group("single_quoted_value")
        or match.group("unquoted_value")
    )


def _authorization_secret(value: str) -> str:
    """Strip a recognized auth scheme before applying placeholder rules."""
    scheme_and_value = value.strip().split(maxsplit=1)
    if len(scheme_and_value) == 2 and scheme_and_value[0].lower() in {
        "basic",
        "bearer",
        "token",
    }:
        return scheme_and_value[1]
    return value


def scan_text(path: str, text: str) -> list[Finding]:
    """Return non-secret-bearing descriptions of suspicious lines."""
    findings: list[Finding] = []
    seen: set[tuple[int, str]] = set()

    def add(line_number: int, rule: str) -> None:
        key = (line_number, rule)
        if key not in seen:
            seen.add(key)
            findings.append(Finding(path, line_number, rule))

    for line_number, line in enumerate(text.splitlines(), start=1):
        for rule, pattern in TOKEN_RULES:
            if pattern.search(line):
                add(line_number, rule)

        for match in CREDENTIALED_URL.finditer(line):
            host = match.group("host").lower().rstrip(".")
            example_url = (
                host.endswith(".invalid")
                and _is_placeholder(match.group("user"))
                and _is_placeholder(match.group("password"))
            )
            if not example_url:
                add(line_number, "url-userinfo")

        for match in GENERIC_ASSIGNMENT.finditer(line):
            if not _is_placeholder(_match_value(match)):
                add(line_number, "literal-secret-assignment")

        authorization_spans: list[tuple[int, int]] = []
        for match in AUTHORIZATION_ASSIGNMENT.finditer(line):
            authorization_spans.append(match.span())
            value = _authorization_secret(_match_value(match))
            if not _is_placeholder(value):
                add(line_number, "authorization-token")

        for match in BEARER_TOKEN.finditer(line):
            belongs_to_authorization = any(
                start <= match.start() < end for start, end in authorization_spans
            )
            if not belongs_to_authorization and not _is_placeholder(_match_value(match)):
                add(line_number, "bearer-token")

    return findings


def _git_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
    )
    paths = [root / os.fsdecode(item) for item in result.stdout.split(b"\0") if item]
    return [path for path in paths if path.exists() or path.is_symlink()]


def _read_candidate(path: Path) -> tuple[str | None, str | None]:
    if path.is_symlink():
        return os.readlink(path), None
    try:
        raw = read_bounded_regular_bytes(
            path.absolute(), MAX_FILE_BYTES, "scan candidate", allow_empty=True
        )
    except ValueError as exc:
        if "must contain 0.." in str(exc) or "exceeds" in str(exc):
            return None, "file-too-large"
        raise
    return raw.decode("utf-8", errors="replace"), None


def _safe_location(value: str) -> str:
    # Paths are attacker-controlled input too. A credential can be placed in a
    # perfectly printable filename and would otherwise be echoed when a finding
    # in that file is reported. Apply the same conservative rules to the display
    # value and replace suspicious or unsafe paths with a stable opaque digest.
    path_contains_secret = bool(scan_text("<path>", value))
    if (
        len(value) > 512
        or any(not character.isprintable() for character in value)
        or path_contains_secret
    ):
        digest = hashlib.sha256(value.encode("utf-8", errors="surrogateescape")).hexdigest()
        return f"worktree-path/{digest}"
    return value


def scan_paths(root: Path, paths: Iterable[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(paths):
        try:
            relative = _safe_location(str(path.relative_to(root)))
            text, error = _read_candidate(path)
        except (OSError, ValueError) as exc:
            relative = _safe_location(str(path))
            findings.append(Finding(relative, 0, f"unreadable:{type(exc).__name__}"))
            continue
        if error is not None:
            findings.append(Finding(relative, 0, error))
        elif text is not None:
            findings.extend(scan_text(relative, text))
    return findings


def _resolve_commit(root: Path, revision: str) -> str:
    if not revision or "\0" in revision or any(char.isspace() for char in revision):
        raise ValueError("invalid Git revision")
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{revision}^{{commit}}",
        ],
        check=True,
        capture_output=True,
    )
    commit = result.stdout.strip().decode("ascii")
    if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", commit):
        raise ValueError("Git returned an invalid object ID")
    return commit.lower()


def _is_zero_object_id(revision: str) -> bool:
    return len(revision) in {40, 64} and not revision.strip("0")


def _introduced_object_ids(root: Path, revision_spec: str) -> list[bytes]:
    """Return unique reachable object IDs while bounding Git's streamed output."""
    process = subprocess.Popen(
        [
            "git",
            "-C",
            str(root),
            "rev-list",
            "--objects",
            "--no-object-names",
            "--end-of-options",
            revision_spec,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    output_stream = process.stdout
    if output_stream is None:  # pragma: no cover - Popen contract
        process.kill()
        process.wait()
        raise OSError("unable to open Git revision stream")

    object_ids: set[bytes] = set()
    try:
        while True:
            line = output_stream.readline(_MAX_GIT_OBJECT_ID_LINE_BYTES + 1)
            if not line:
                break
            if len(line) > _MAX_GIT_OBJECT_ID_LINE_BYTES or not line.endswith(b"\n"):
                raise ValueError("Git returned malformed object IDs")
            object_id = line[:-1]
            if not re.fullmatch(rb"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", object_id):
                raise ValueError("Git returned an invalid object ID")
            object_ids.add(object_id.lower())
            if len(object_ids) > MAX_GIT_OBJECTS:
                raise ValueError("Git range exceeds the object-count safety limit")

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, process.args)
    finally:
        output_stream.close()
        if process.poll() is None:
            process.kill()
        process.wait()

    return sorted(object_ids)


def _introduced_object_metadata(root: Path, revision_range: str) -> list[tuple[str, str, int]]:
    if revision_range.count("..") != 1 or "..." in revision_range:
        raise ValueError("Git range must have the form BASE..HEAD")
    base_revision, head_revision = revision_range.split("..", maxsplit=1)
    head = _resolve_commit(root, head_revision)
    if _is_zero_object_id(base_revision):
        # GitHub reports an all-zero `before` SHA for the first push of a ref.
        # Scanning HEAD alone traverses the complete newly reachable history.
        revision_spec = head
    else:
        base = _resolve_commit(root, base_revision)
        revision_spec = f"{base}..{head}"

    object_ids = _introduced_object_ids(root, revision_spec)
    if not object_ids:
        return []

    metadata = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "cat-file",
            "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        ],
        input=b"\n".join(object_ids) + b"\n",
        check=True,
        capture_output=True,
    )

    metadata_lines = metadata.stdout.splitlines()
    if len(metadata_lines) != len(object_ids):
        raise ValueError("Git returned incomplete object metadata")

    candidates: list[tuple[str, str, int]] = []
    readable_bytes = 0
    for requested_id, line in zip(object_ids, metadata_lines, strict=True):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError("Git returned malformed object metadata")
        object_id, object_type, raw_size = fields
        if object_id.lower() != requested_id:
            raise ValueError("Git returned unexpected object metadata")
        decoded_type = object_type.decode("ascii")
        if decoded_type not in {"blob", "commit", "tag"}:
            continue
        decoded_id = object_id.decode("ascii").lower()
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", decoded_id):
            raise ValueError("Git returned an invalid object ID")
        size = int(raw_size)
        if size < 0:
            raise ValueError("Git returned an invalid object size")
        if size <= MAX_FILE_BYTES:
            readable_bytes += size
            if readable_bytes > MAX_GIT_SCAN_BYTES:
                raise ValueError("Git range exceeds the cumulative-byte safety limit")
        candidates.append((decoded_id, decoded_type, size))
    return candidates


def scan_git_range(root: Path, revision_range: str) -> tuple[list[Finding], int]:
    """Scan introduced blobs and commit/tag messages without printing contents."""
    findings: list[Finding] = []
    objects = _introduced_object_metadata(root, revision_range)
    readable_objects: list[tuple[str, str, int]] = []
    for object_id, object_type, size in objects:
        location = f"git-{object_type}/{object_id}"
        if size > MAX_FILE_BYTES:
            findings.append(Finding(location, 0, "git-object-too-large"))
        else:
            readable_objects.append((object_id, object_type, size))

    if not readable_objects:
        return findings, len(objects)

    process = subprocess.Popen(
        ["git", "-C", str(root), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    input_stream = process.stdin
    output_stream = process.stdout
    if input_stream is None or output_stream is None:  # pragma: no cover - Popen contract
        process.kill()
        process.wait()
        raise OSError("unable to open Git object stream")

    try:
        for object_id, expected_type, expected_size in readable_objects:
            input_stream.write(object_id.encode("ascii") + b"\n")
            input_stream.flush()

            header = output_stream.readline().split()
            if len(header) != 3:
                raise ValueError("Git returned malformed blob metadata")
            returned_id, object_type, raw_size = header
            size = int(raw_size)
            if (
                returned_id.decode("ascii").lower() != object_id
                or object_type.decode("ascii") != expected_type
                or size != expected_size
            ):
                raise ValueError("Git returned unexpected blob metadata")

            raw = output_stream.read(size)
            delimiter = output_stream.read(1)
            location = f"git-{expected_type}/{object_id}"
            if len(raw) != size or delimiter != b"\n":
                findings.append(Finding(location, 0, "blob-size-mismatch"))
            else:
                findings.extend(scan_text(location, raw.decode("utf-8", errors="replace")))

        input_stream.close()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, process.args)
    finally:
        if not input_stream.closed:
            input_stream.close()
        if process.poll() is None:
            process.kill()
        process.wait()
        output_stream.close()

    return findings, len(objects)


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", type=Path, help="explicit files (default: Git worktree)"
    )
    parser.add_argument(
        "--git-range",
        metavar="BASE..HEAD",
        help="scan every Git blob introduced across a two-dot revision range",
    )
    args = parser.parse_args(argv)

    root = (root or Path.cwd()).resolve()
    if args.git_range is not None and args.paths:
        parser.error("explicit paths cannot be combined with --git-range")

    try:
        if args.git_range is not None:
            findings, checked = scan_git_range(root, args.git_range)
            subject = "Git objects"
        else:
            paths = [path.resolve() for path in args.paths] if args.paths else _git_files(root)
            findings = scan_paths(root, paths)
            checked = len(paths)
            subject = "files"
    except (OSError, UnicodeError, ValueError, subprocess.CalledProcessError):
        print("secret scan failed: unable to read the requested Git content", file=sys.stderr)
        return 2

    if findings:
        for finding in findings:
            print(
                f"secret scan finding: {finding.path}:{finding.line} ({finding.rule})",
                file=sys.stderr,
            )
        return 1
    print(f"secret scan passed: {checked} {subject} checked")
    return 0
