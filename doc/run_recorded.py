#!/usr/bin/env python3
"""Run the documentation check in a fresh, self-contained results directory."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import traceback


class CheckFailure(RuntimeError):
    pass


class RecordedRun:
    def __init__(self, repo, directory):
        self.repo = Path(repo).resolve()
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = {"started_utc": datetime.now(timezone.utc).isoformat(),
                         "commands": [], "exit_status": None}
        self.save()

    def save(self):
        (self.directory / "metadata.json").write_text(json.dumps(self.metadata, indent=2) + "\n")

    def command(self, label, command, *, required=True, env=None):
        command = [str(part) for part in command]
        record = {"label": label, "argv": command, "cwd": str(self.repo), "exit_status": None,
                  "log": f"{label}.log"}
        self.metadata["commands"].append(record)
        self.save()
        print(f"{label}: {' '.join(command)}", flush=True)
        with (self.directory / record["log"]).open("w") as log:
            try:
                result = subprocess.run(command, cwd=self.repo, env=env, stdout=log, stderr=subprocess.STDOUT)
                record["exit_status"] = result.returncode
            except OSError as error:
                log.write(f"{type(error).__name__}: {error}\n")
                record["exit_status"] = 127
        self.save()
        print(f"{label}: exit {record['exit_status']} ({self.directory / record['log']})", flush=True)
        if required and record["exit_status"]:
            raise CheckFailure(f"{label} failed with exit {record['exit_status']}")
        return (self.directory / record["log"]).read_text()


def stage_sources(repo, stage):
    shutil.copytree(repo / "doc/source", stage / "doc/source", ignore=shutil.ignore_patterns("verilog", "autoapi", "__pycache__"))
    shutil.copytree(repo / "doc/figures", stage / "doc/figures")
    for source in (repo / "test").rglob("*.py"):
        if not any(part in {"__pycache__", ".venv", "venv"} for part in source.parts):
            target = stage / source.relative_to(repo)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def check_sphinx_log(log):
    # Sphinx can report an error and still exit with status 0.  
    diagnostics = re.findall(r"(?m)^.*?\b(DEBUG|INFO|WARNING|ERROR|SEVERE)(?::|/[0-9]+\))", log)
    if any(level in {"ERROR", "SEVERE"} for level in diagnostics) or re.search(r"Traceback \(most recent call last\)", log):
        raise CheckFailure("Sphinx emitted an ERROR/SEVERE diagnostic; see sphinx.log")


def write_manifest(directory, report):
    files = {}
    for area in ("stage", "html"):
        for path in sorted((directory / area).rglob("*")):
            if path.is_file():
                files[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"status": "success", "coverage": report, "sha256": files}
    # Publish the success manifest only after the write completes
    temporary = directory / "manifest.pending"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(directory / "manifest.json")


def perform(run, args):
    repo, directory = run.repo, run.directory
    run.metadata["configuration"] = {"profile": "structured-comments", "allow_partial": args.allow_partial,
                                     "diagrams": "required", "python": args.python,
                                     "defines": ["USE_INTERNAL_PARAMS", "USE_INTERNAL_PARAMS_PE", "USE_INTERNAL_PARAMS_PE_cluster"]}
    run.command("revision", ["git", "rev-parse", "HEAD"])
    run.command("branch", ["git", "branch", "--show-current"])
    run.command("remotes", ["git", "remote", "-v"])
    run.command("working-tree", ["git", "status", "--short", "--untracked-files=all"])
    run.command("source-patch", ["git", "diff", "--binary", "HEAD"])
    config = directory / "configuration"
    config.mkdir()
    for source in [repo / "doc/run_recorded.py", repo / "doc/validate_output.py",
                   repo / "doc/requirements.txt", repo / "doc/requirements.lock",
                   repo / "doc/source/conf.py", repo / ".github/workflows/documentation.yml",
                   repo / ".gitlab-ci.yml",
                   repo / ".readthedocs.yaml", repo / "doc/documentation-check.md",
                   *sorted((repo / "doc").glob("test_*.py"))]:
        if source.exists():
            target = config / source.relative_to(repo)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    shutil.copytree(repo / "doc/verilog_doc_parser", config / "doc/verilog_doc_parser",
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    run.command("platform", ["uname", "-a"])
    run.command("native-packages", ["dpkg-query", "-W", "build-essential", "pkg-config", "python3.12-dev",
                                    "libcairo2-dev", "libgirepository1.0-dev", "gir1.2-pango-1.0"], required=False)
    python = Path(args.python).absolute()
    run.command("base-python-version", [python, "--version"])
    if args.install:
        environment = directory / "environment"
        run.command("venv", [python, "-m", "venv", environment])
        python = environment / "bin/python"
        run.command("bootstrap", [python, "-m", "pip", "install", "-c", repo / "doc/requirements.lock",
                                  "pip==26.2.1", "setuptools==84.0.0", "wheel==0.48.0"])
        run.command("install", [python, "-m", "pip", "install", "--no-cache-dir", "--build-constraint", repo / "doc/requirements.lock",
                                "-r", repo / "doc/requirements.txt"])
    run.command("python-version", [python, "--version"])
    run.command("python-profile", [python, "-c", "import sys; assert sys.version_info[:2] == (3, 12), sys.version"])
    run.command("dependencies", [python, "-m", "pip", "freeze", "--all"])
    run.command("pip-check", [python, "-m", "pip", "check"])
    env = os.environ.copy()
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    symbolator = args.symbolator or str(python.parent / "symbolator")
    sphinx = args.sphinx or str(python.parent / "sphinx-build")
    run.command("symbolator-version", [symbolator, "--version"], env=env)
    run.command("sphinx-version", [sphinx, "--version"], env=env)
    run.command("tests", [python, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           "--basetemp", directory / "test-results",
                           repo / "doc/verilog_doc_parser", repo / "doc/test_validate_output.py",
                           repo / "doc/test_run_recorded.py", repo / "doc/test_yosys_resources.py"], env=env)
    stage = directory / "stage"
    stage_sources(repo, stage)
    generated = stage / "doc/source/verilog"
    command = [python, repo / "doc/verilog_doc_parser/rst_doc_generator.py", "--output-dir", generated,
               "--hdl-dir", repo / "hdl", "--symbolator", symbolator]
    if args.allow_partial:
        command.append("--allow-partial")
    run.command("generate", command, env=env)
    report = json.loads((generated / "coverage.json").read_text())
    shutil.copy2(generated / "coverage.json", directory / "coverage.json")
    log = run.command("sphinx", [sphinx, "-b", "html", "-E", "-a", "-T", "-d", directory / "doctrees",
                                stage / "doc/source", directory / "html"], env=env)
    check_sphinx_log(log)
    run.command("validate", [python, repo / "doc/validate_output.py", "--generated-dir", generated,
                             "--html-dir", directory / "html", "--report", directory / "validation.json"], env=env)
    run.metadata["exit_status"] = 0
    run.save()
    write_manifest(directory, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path(tempfile.gettempdir()) / "openeye-documentation-results")
    parser.add_argument("--run-dir", type=Path, help="Explicit fresh result directory, useful for automation")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--install", action="store_true", help="Install pinned dependencies into a new per-run venv")
    parser.add_argument("--allow-partial", action="store_true", help="Accept modules lacking structured comments; report every omission")
    parser.add_argument("--symbolator", help="Override executable for controlled failure checks")
    parser.add_argument("--sphinx", help="Override executable for controlled failure checks")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    args.results_root.mkdir(parents=True, exist_ok=True)
    directory = args.run_dir or args.results_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{os.getpid()}")
    try:
        run = RecordedRun(repo, directory)
    except OSError as error:
        print(f"Cannot create fresh results: {error}", file=sys.stderr)
        return 1
    print(f"Results: {run.directory}", flush=True)
    try:
        perform(run, args)
    except Exception as error:
        run.metadata["exit_status"] = 1
        run.metadata["failure"] = f"{type(error).__name__}: {error}"
        (run.directory / "failure.log").write_text(traceback.format_exc())
        run.save()
        print(run.metadata["failure"], file=sys.stderr)
        return 1
    print(f"Documentation check passed: {run.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
