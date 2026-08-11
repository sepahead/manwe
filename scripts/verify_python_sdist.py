#!/usr/bin/env python3
"""Verify that a Python sdist is a bounded archive of tracked release inputs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
from typing import IO

_MAX_MEMBERS = 100_000
_MAX_ARCHIVE_BYTES = 256 << 20
_MAX_UNCOMPRESSED_BYTES = 1 << 30
_MAX_GENERATED_METADATA_BYTES = 2 << 20
_SAFE_ARCHIVE_MODES = frozenset({0o644, 0o755})
_PROJECT_INPUTS = frozenset({"LICENSE", "README.md", "pyproject.toml", "uv.lock"})
_PROJECT_TREES = frozenset({"configs", "src", "tests"})


def _sha256_stream(stream: IO[bytes]) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1 << 20):
        digest.update(chunk)
    return digest.hexdigest()


def _tracked_inputs(repo_root: pathlib.Path, project_root: pathlib.Path) -> set[str]:
    try:
        project_relative = project_root.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("Python project root must be inside the repository root") from exc
    selectors = [
        ".gitignore",
        *(str(project_relative / name) for name in sorted(_PROJECT_INPUTS)),
        *(str(project_relative / name) for name in sorted(_PROJECT_TREES)),
    ]
    result = subprocess.run(
        ["git", "-C", os.fspath(repo_root), "ls-files", "-z", "--", *selectors],
        check=True,
        stdout=subprocess.PIPE,
    )
    repository_paths = {
        pathlib.PurePosixPath(os.fsdecode(value)) for value in result.stdout.split(b"\0") if value
    }
    expected: set[str] = set()
    project_parts = pathlib.PurePosixPath(project_relative.as_posix()).parts
    for repository_path in repository_paths:
        if repository_path == pathlib.PurePosixPath(".gitignore"):
            expected.add(".gitignore")
            continue
        if repository_path.parts[: len(project_parts)] != project_parts:
            continue
        local = pathlib.PurePosixPath(*repository_path.parts[len(project_parts) :])
        if str(local) in _PROJECT_INPUTS or (local.parts and local.parts[0] in _PROJECT_TREES):
            expected.add(local.as_posix())
    required = _PROJECT_INPUTS | {".gitignore"}
    if missing := required - expected:
        raise ValueError(f"release inputs are not tracked: {sorted(missing)}")
    expected.add("PKG-INFO")
    return expected


def verify_sdist(
    archive_path: pathlib.Path,
    *,
    repo_root: pathlib.Path,
    project_root: pathlib.Path,
) -> tuple[str, int, int]:
    """Return archive root, member count, and bytes after strict verification."""
    archive_path = archive_path.resolve(strict=True)
    repo_root = repo_root.resolve(strict=True)
    project_root = project_root.resolve(strict=True)
    archive_size = archive_path.stat().st_size
    if not 1 <= archive_size <= _MAX_ARCHIVE_BYTES:
        raise ValueError(f"sdist archive must contain 1..={_MAX_ARCHIVE_BYTES} compressed bytes")
    expected = _tracked_inputs(repo_root, project_root)
    expected_root = archive_path.name.removesuffix(".tar.gz")
    if not expected_root or expected_root == archive_path.name:
        raise ValueError("sdist filename must end in .tar.gz")

    observed: set[str] = set()
    total_bytes = 0
    member_count = 0
    with tarfile.open(archive_path, mode="r:gz") as source:
        for member in source:
            member_count += 1
            if member_count > _MAX_MEMBERS:
                raise ValueError(f"sdist exceeds the {_MAX_MEMBERS}-member safety limit")
            raw_parts = member.name.split("/")
            if (
                member.name.startswith("/")
                or "\\" in member.name
                or len(raw_parts) < 2
                or raw_parts[0] != expected_root
                or any(part in {"", ".", ".."} for part in raw_parts)
            ):
                raise ValueError(f"sdist contains an unsafe member path: {member.name!r}")
            if not member.isfile():
                raise ValueError(f"sdist contains a non-regular member: {member.name!r}")
            if member.size < 0:
                raise ValueError(f"sdist contains a negative-size member: {member.name!r}")
            if member.mode not in _SAFE_ARCHIVE_MODES:
                raise ValueError(
                    f"sdist contains unsafe regular-file mode {member.mode:#o}: {member.name!r}"
                )
            if member.uid != 0 or member.gid != 0 or member.uname or member.gname:
                raise ValueError(f"sdist contains non-canonical ownership: {member.name!r}")
            if member.pax_headers:
                raise ValueError(f"sdist contains unmodeled PAX metadata: {member.name!r}")
            relative = pathlib.PurePosixPath(*raw_parts[1:]).as_posix()
            if relative in observed:
                raise ValueError(f"sdist contains a duplicate member: {relative!r}")
            observed.add(relative)
            if relative not in expected:
                raise ValueError(f"sdist contains an untracked release input: {relative!r}")
            total_bytes += member.size
            if total_bytes > _MAX_UNCOMPRESSED_BYTES:
                raise ValueError(
                    f"sdist exceeds the {_MAX_UNCOMPRESSED_BYTES}-byte uncompressed safety limit"
                )

            if relative == "PKG-INFO":
                if member.size > _MAX_GENERATED_METADATA_BYTES:
                    raise ValueError("sdist PKG-INFO exceeds the generated-metadata safety limit")
                continue
            source_path = (
                repo_root / ".gitignore"
                if relative == ".gitignore"
                else project_root / pathlib.PurePosixPath(relative)
            )
            if not source_path.is_file() or source_path.is_symlink():
                raise ValueError(f"tracked release input is not a regular file: {relative!r}")
            archived = source.extractfile(member)
            if archived is None:
                raise ValueError(f"sdist member cannot be read: {relative!r}")
            with archived, source_path.open("rb") as current:
                if _sha256_stream(archived) != _sha256_stream(current):
                    raise ValueError(
                        f"sdist member differs from the current tracked input: {relative!r}"
                    )

    if member_count == 0:
        raise ValueError("sdist must contain at least one member")

    if missing := expected - observed:
        raise ValueError(f"sdist is missing tracked release inputs: {sorted(missing)}")
    if unexpected := observed - expected:
        raise ValueError(f"sdist contains untracked release inputs: {sorted(unexpected)}")
    return expected_root, member_count, total_bytes


def _write_mutant(
    source_path: pathlib.Path,
    destination_path: pathlib.Path,
    *,
    mutation: str,
) -> None:
    with tarfile.open(source_path, mode="r:gz") as source:
        members = source.getmembers()
        root = members[0].name.split("/", 1)[0]
        with tarfile.open(destination_path, mode="w:gz") as destination:
            for member in members:
                archived_payload = source.extractfile(member)
                if archived_payload is None:
                    raise ValueError(f"verified source member cannot be read: {member.name!r}")
                with archived_payload:
                    destination.addfile(member, archived_payload)
            if mutation == "special":
                probe = tarfile.TarInfo(f"{root}/tests/UNTRACKED_FIFO")
                probe.type = tarfile.FIFOTYPE
                destination.addfile(probe)
            elif mutation == "untracked":
                probe_payload = b"untracked workstation state"
                probe = tarfile.TarInfo(f"{root}/tests/UNTRACKED_SENTINEL")
                probe.size = len(probe_payload)
                destination.addfile(probe, io.BytesIO(probe_payload))
            elif mutation == "mode":
                probe = copy.copy(members[0])
                probe.mode = 0o666
                archived_payload = source.extractfile(members[0])
                if archived_payload is None:
                    raise ValueError(f"verified source member cannot be read: {members[0].name!r}")
                with archived_payload:
                    destination.addfile(probe, archived_payload)
            else:  # pragma: no cover - internal caller invariant
                raise ValueError(f"unknown sdist mutation: {mutation}")


def _adversarial_self_test(
    archive_path: pathlib.Path,
    *,
    repo_root: pathlib.Path,
    project_root: pathlib.Path,
) -> None:
    with tempfile.TemporaryDirectory(prefix="manwe-sdist-verifier-") as temporary:
        temporary_root = pathlib.Path(temporary)
        for label, expected_message in (
            ("untracked", "untracked release input"),
            ("special", "non-regular member"),
            ("mode", "unsafe regular-file mode"),
        ):
            destination_dir = temporary_root / label
            destination_dir.mkdir()
            mutant = destination_dir / archive_path.name
            _write_mutant(archive_path, mutant, mutation=label)
            try:
                verify_sdist(mutant, repo_root=repo_root, project_root=project_root)
            except ValueError as exc:
                if expected_message not in str(exc):
                    raise ValueError(
                        f"{label} sdist mutant failed for the wrong reason: {exc}"
                    ) from exc
            else:
                raise ValueError(f"{label} sdist mutant was incorrectly accepted")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=pathlib.Path)
    repository_default = pathlib.Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-root", type=pathlib.Path, default=repository_default)
    parser.add_argument(
        "--project-root",
        type=pathlib.Path,
        default=repository_default / "python",
    )
    parser.add_argument(
        "--adversarial-self-test",
        action="store_true",
        help="also prove rejection of nested untracked and special archive members",
    )
    args = parser.parse_args(argv)
    try:
        root, members, total_bytes = verify_sdist(
            args.archive,
            repo_root=args.repo_root,
            project_root=args.project_root,
        )
        if args.adversarial_self_test:
            _adversarial_self_test(
                args.archive.resolve(strict=True),
                repo_root=args.repo_root,
                project_root=args.project_root,
            )
    except (OSError, ValueError, subprocess.CalledProcessError, tarfile.TarError) as exc:
        parser.exit(1, f"sdist verification failed: {exc}\n")
    suffix = "; adversarial mutants rejected" if args.adversarial_self_test else ""
    print(f"verified bounded sdist {root}: {members} files, {total_bytes} bytes{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
