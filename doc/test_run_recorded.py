"""Failure boundaries for the local/CI launcher; no shared output is touched."""
import json
from pathlib import Path
import pytest

from doc import run_recorded as runner


def test_fresh_results_refuse_previous_success(tmp_path):
    previous = tmp_path / "success"
    previous.mkdir()
    manifest = previous / "manifest.json"
    manifest.write_text('{"status": "success"}')
    with pytest.raises(FileExistsError):
        runner.RecordedRun(tmp_path, previous)
    assert manifest.read_text() == '{"status": "success"}'


def test_command_failure_records_real_exit_and_no_manifest(tmp_path):
    run = runner.RecordedRun(tmp_path, tmp_path / "failed")
    with pytest.raises(runner.CheckFailure, match="controlled failed with exit 7"):
        run.command("controlled", ["/bin/sh", "-c", "printf diagnostic; exit 7"])
    saved = json.loads((run.directory / "metadata.json").read_text())
    assert saved["commands"][0]["exit_status"] == 7
    assert (run.directory / "controlled.log").read_text() == "diagnostic"
    assert not (run.directory / "manifest.json").exists()


def test_zero_exit_error_diagnostic_is_fatal():
    runner.check_sphinx_log("file.rst:1: WARNING: unknown target name: ERROR")
    with pytest.raises(runner.CheckFailure, match="ERROR/SEVERE"):
        runner.check_sphinx_log("file.rst:1: ERROR: invalid directive")


def test_manifest_write_failure_cannot_be_success(tmp_path, monkeypatch):
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "manifest.json").write_text("previous success")
    current = tmp_path / "current"
    current.mkdir()
    original = Path.write_text

    def fail_manifest(path, *args, **kwargs):
        if path.name == "manifest.pending":
            raise PermissionError("controlled manifest failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_manifest)
    with pytest.raises(PermissionError, match="controlled manifest failure"):
        runner.write_manifest(current, {})
    assert not (current / "manifest.json").exists()
    assert (previous / "manifest.json").read_text() == "previous success"


def test_main_build_failure_preserves_diagnostics(tmp_path, monkeypatch):
    def fail_build(run, args):
        run.command("sphinx", ["/bin/sh", "-c", "printf build-failure; exit 3"])

    monkeypatch.setattr(runner, "perform", fail_build)
    directory = tmp_path / "build-failed"
    assert runner.main(["--run-dir", str(directory), "--results-root", str(tmp_path)]) == 1
    assert json.loads((directory / "metadata.json").read_text())["exit_status"] == 1
    assert (directory / "sphinx.log").read_text() == "build-failure"
    assert "sphinx failed with exit 3" in (directory / "failure.log").read_text()
    assert not (directory / "manifest.json").exists()
