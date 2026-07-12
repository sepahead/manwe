"""CLI errors distinguish expected user failures from unexpected defects."""

from __future__ import annotations

import pytest

import manwe.cli as cli


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["data", "not-a-dataset"], "unknown dataset"),
        (["data", "--modality", "sonar"], "invalid choice"),
        (["synth", "unused", "--n-train", "0"], "n-train must be >= 1"),
        (["fusion-sim", "--duration", "1"], "duration must be finite and >= 1.5"),
        (["fusion-sim", "--p-detect", "2"], "p-detect must be finite and in [0, 1]"),
        (["fusion-sim", "--modalities", "sonar"], "invalid choice"),
        (
            [
                "export",
                "weights.pt",
                "--format",
                "onnx",
                "--output",
                "model.onnx",
                "--weights-sha256",
                "not-a-digest",
                "--allow-unverified",
            ],
            "weights-sha256 must be a 64-character hexadecimal digest",
        ),
        (
            [
                "export",
                "weights.pt",
                "--format",
                "onnx",
                "--output",
                "model.onnx",
                "--weights-sha256",
                "a" * 64,
            ],
            "raw export is not a trusted consumer handoff",
        ),
    ],
)
def test_expected_input_errors_are_concise_argparse_failures(argv, message, capsys):
    with pytest.raises(SystemExit) as raised:
        cli.main(argv)

    assert raised.value.code == 2
    stderr = capsys.readouterr().err
    assert "usage: manwe" in stderr
    assert ": error:" in stderr
    assert message in stderr
    assert "Traceback" not in stderr


def test_missing_config_is_a_concise_argparse_failure(tmp_path, capsys):
    missing = tmp_path / "missing.yaml"

    with pytest.raises(SystemExit) as raised:
        cli.main(["vision-train", str(missing)])

    assert raised.value.code == 2
    stderr = capsys.readouterr().err
    assert "vision training config path does not exist" in stderr
    assert "Traceback" not in stderr


def test_missing_export_artifact_is_a_concise_argparse_failure(tmp_path, capsys):
    missing = tmp_path / "missing.pt"
    output = tmp_path / "model.onnx"

    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                "export",
                str(missing),
                "--format",
                "onnx",
                "--output",
                str(output),
                "--weights-sha256",
                "a" * 64,
                "--allow-pickle-checkpoint",
                "--allow-unverified",
            ]
        )

    assert raised.value.code == 2
    stderr = capsys.readouterr().err
    assert "No such file or directory" in stderr
    assert "Traceback" not in stderr


def test_unexpected_command_errors_still_propagate(monkeypatch):
    def fail_unexpectedly(_args):
        raise RuntimeError("internal sentinel")

    monkeypatch.setattr(cli, "_cmd_data", fail_unexpectedly)

    with pytest.raises(RuntimeError, match="internal sentinel"):
        cli.main(["data"])
