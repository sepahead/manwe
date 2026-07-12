import subprocess
from pathlib import Path

import pytest

from manwe.common import secret_scan
from manwe.common.secret_scan import main, scan_paths, scan_text


def test_secret_scanner_reports_location_without_value() -> None:
    credential_url = "rtsp://operator:" + "sensitive-value-123" + "@camera.example/live"

    findings = scan_text("camera.conf", credential_url)

    assert [(item.path, item.line, item.rule) for item in findings] == [
        ("camera.conf", 1, "url-userinfo")
    ]
    assert "sensitive-value-123" not in repr(findings)


def test_secret_scanner_allows_non_routable_placeholder_url() -> None:
    assert scan_text("example.env", "rtsp://user:password@camera.invalid/live") == []


def test_secret_scanner_does_not_allow_placeholder_substrings_on_live_hosts() -> None:
    credential_url = "rtsp://operator:" + "real-example-password" + "@camera.example/live"

    findings = scan_text("camera.conf", credential_url)

    assert [item.rule for item in findings] == ["url-userinfo"]


@pytest.mark.parametrize(
    "quoted",
    [True, False],
)
@pytest.mark.parametrize("key", ["password", "client_secret"])
def test_secret_scanner_reports_quoted_and_unquoted_assignments(quoted: bool, key: str) -> None:
    credential = "sensitive" + "-value-456"
    value = f'"{credential}"' if quoted else credential

    findings = scan_text("service.env", f"{key}={value}")

    assert [item.rule for item in findings] == ["literal-secret-assignment"]
    assert credential not in repr(findings)


def test_secret_scanner_requires_the_entire_value_to_be_a_placeholder() -> None:
    key = "client" + "_secret"
    placeholder = "${CLIENT_SECRET}"

    assert scan_text("service.env", f'{key}="{placeholder}"') == []
    findings = scan_text("service.env", f'{key}="{placeholder}-real-value"')

    assert [item.rule for item in findings] == ["literal-secret-assignment"]


def test_secret_scanner_reports_authorization_and_bearer_tokens() -> None:
    header_name = "Authori" + "zation"
    auth_scheme = "Bear" + "er"
    token = "opaque" + "-credential-789"

    bearer_findings = scan_text("headers.txt", f"{header_name}: {auth_scheme} {token}")
    standalone_findings = scan_text("headers.txt", f"{auth_scheme} {token}")
    direct_findings = scan_text("headers.txt", f'{header_name}="{token}"')

    assert [item.rule for item in bearer_findings] == ["authorization-token"]
    assert [item.rule for item in standalone_findings] == ["bearer-token"]
    assert [item.rule for item in direct_findings] == ["authorization-token"]
    assert token not in repr(bearer_findings + standalone_findings + direct_findings)


def test_secret_scanner_handles_the_opposite_quote_inside_quoted_values() -> None:
    key = "pass" + "word"
    credential = "owner's" + "-credential"

    findings = scan_text("service.toml", f'{key}="{credential}"')

    assert [item.rule for item in findings] == ["literal-secret-assignment"]


@pytest.mark.parametrize(
    "key",
    ["AWS_SECRET_ACCESS_KEY", "OPENAI_API_KEY", "HF_TOKEN", "GITLAB_TOKEN", "NPM_TOKEN"],
)
def test_secret_scanner_reports_namespaced_secret_assignments(key: str) -> None:
    credential = "namespaced" + "-credential-123456789"
    findings = scan_text("service.env", f"{key}={credential}")
    assert [item.rule for item in findings] == ["literal-secret-assignment"]
    assert credential not in repr(findings)


def test_secret_scanner_ignores_unrecognized_nonsecret_token_key() -> None:
    assert scan_text("config.env", "NOT_TOKEN=public") == []


@pytest.mark.parametrize(
    ("prefix", "rule"),
    [
        ("sk-proj-", "openai-token"),
        ("hf_", "huggingface-token"),
        ("glpat-", "gitlab-token"),
        ("npm_", "npm-token"),
        ("pypi-", "pypi-token"),
    ],
)
def test_secret_scanner_reports_standalone_provider_tokens(prefix: str, rule: str) -> None:
    credential = prefix + "A" * 40
    findings = scan_text("token.txt", credential)
    assert [item.rule for item in findings] == [rule]
    assert credential not in repr(findings)


def test_worktree_scan_checks_bounded_binary_content(tmp_path: Path) -> None:
    credential = "binary" + "-credential-current-worktree"
    path = tmp_path / "payload.bin"
    path.write_bytes(b"\0" + f"{'pass' + 'word'}={credential}".encode())
    findings = scan_paths(tmp_path, [path])
    assert [item.rule for item in findings] == ["literal-secret-assignment"]
    assert credential not in repr(findings)


@pytest.mark.parametrize(
    "scheme",
    [
        "postgres",
        "postgresql",
        "mysql",
        "mongodb",
        "mongodb+srv",
        "redis",
        "amqp",
    ],
)
def test_secret_scanner_reports_database_and_broker_url_credentials(scheme: str) -> None:
    credential = "database" + "-credential-123"
    url = f"{scheme}://service:{credential}@db.example/data"

    findings = scan_text("service.env", url)

    assert [item.rule for item in findings] == ["url-userinfo"]
    assert credential not in repr(findings)


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_git_range_scans_transient_blobs_without_echoing_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.email", "scanner@example.invalid")
    _git(tmp_path, "config", "user.name", "Scanner Test")
    _git(tmp_path, "config", "commit.gpgsign", "false")

    legacy_value = "legacy" + "-credential-should-be-excluded"
    (tmp_path / "legacy.env").write_text(f'{"pass" + "word"}="{legacy_value}"\n', encoding="utf-8")
    _git(tmp_path, "add", "legacy.env")
    _git(tmp_path, "commit", "--quiet", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")

    transient_value = "transient" + "-credential-must-be-found"
    (tmp_path / "transient.env").write_text(
        f"{'client' + '_secret'}={transient_value}\n", encoding="utf-8"
    )
    binary_credential = "binary" + "-credential-must-be-found"
    (tmp_path / "transient.bin").write_bytes(
        b"\0" + f"{'pass' + 'word'}={binary_credential}".encode()
    )
    _git(tmp_path, "add", "transient.env", "transient.bin")
    _git(tmp_path, "commit", "--quiet", "-m", "introduce transient blob")
    (tmp_path / "transient.env").unlink()
    (tmp_path / "transient.bin").unlink()
    _git(tmp_path, "add", "--update")
    _git(tmp_path, "commit", "--quiet", "-m", "remove transient blob")
    head = _git(tmp_path, "rev-parse", "HEAD")

    result = main(["--git-range", f"{base}..{head}"], root=tmp_path)
    captured = capsys.readouterr()

    assert result == 1
    assert "literal-secret-assignment" in captured.err
    assert "git-blob/" in captured.err
    assert transient_value not in captured.err
    assert binary_credential not in captured.err
    assert legacy_value not in captured.err
    assert transient_value not in captured.out
    assert binary_credential not in captured.out
    assert legacy_value not in captured.out


def test_git_range_scans_commit_messages_without_echoing_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.email", "scanner@example.invalid")
    _git(tmp_path, "config", "user.name", "Scanner Test")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "README").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "README")
    _git(tmp_path, "commit", "--quiet", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")

    credential = "message" + "-credential-must-be-found"
    message_key = "client" + "_secret"
    (tmp_path / "README").write_text("next\n", encoding="utf-8")
    _git(tmp_path, "add", "README")
    _git(tmp_path, "commit", "--quiet", "-m", f"{message_key}={credential}")
    head = _git(tmp_path, "rev-parse", "HEAD")

    result = main(["--git-range", f"{base}..{head}"], root=tmp_path)
    captured = capsys.readouterr()
    assert result == 1
    assert "git-commit/" in captured.err
    assert "literal-secret-assignment" in captured.err
    assert credential not in captured.err
    assert credential not in captured.out


def test_git_range_bounds_object_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.email", "scanner@example.invalid")
    _git(tmp_path, "config", "user.name", "Scanner Test")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "README").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "README")
    _git(tmp_path, "commit", "--quiet", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "README").write_text("next\n", encoding="utf-8")
    _git(tmp_path, "add", "README")
    _git(tmp_path, "commit", "--quiet", "-m", "next")
    head = _git(tmp_path, "rev-parse", "HEAD")

    monkeypatch.setattr(secret_scan, "MAX_GIT_OBJECTS", 1)
    with pytest.raises(ValueError, match="object-count safety limit"):
        secret_scan.scan_git_range(tmp_path, f"{base}..{head}")


def test_git_range_bounds_cumulative_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.email", "scanner@example.invalid")
    _git(tmp_path, "config", "user.name", "Scanner Test")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "README").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "README")
    _git(tmp_path, "commit", "--quiet", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "README").write_text("next\n", encoding="utf-8")
    _git(tmp_path, "add", "README")
    _git(tmp_path, "commit", "--quiet", "-m", "next")
    head = _git(tmp_path, "rev-parse", "HEAD")

    monkeypatch.setattr(secret_scan, "MAX_GIT_SCAN_BYTES", 1)
    with pytest.raises(ValueError, match="cumulative-byte safety limit"):
        secret_scan.scan_git_range(tmp_path, f"{base}..{head}")


def test_git_range_accepts_first_push_zero_base_without_echoing_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.email", "scanner@example.invalid")
    _git(tmp_path, "config", "user.name", "Scanner Test")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    credential = "initial" + "-credential-must-be-found"
    (tmp_path / "service.env").write_text(
        f"{'client' + '_secret'}={credential}\n", encoding="utf-8"
    )
    _git(tmp_path, "add", "service.env")
    _git(tmp_path, "commit", "--quiet", "-m", "initial")
    head = _git(tmp_path, "rev-parse", "HEAD")

    result = main(["--git-range", f"{'0' * 40}..{head}"], root=tmp_path)
    captured = capsys.readouterr()

    assert result == 1
    assert "literal-secret-assignment" in captured.err
    assert credential not in captured.err
    assert credential not in captured.out
