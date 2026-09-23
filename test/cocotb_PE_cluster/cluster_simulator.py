"""Reuse Icarus builds within a pytest process; keep each case's artifacts separate."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import cocotb_test.simulator
from open_eye import hdl_dir, test_dir, vh_file_creator


def run_cluster_simulation(**kwargs):
    case_dir = Path(kwargs.pop("sim_build")).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    env = dict(kwargs.get("extra_env", {}))
    parameters = kwargs.get("parameters", {})
    # Explicit defaults prevent a preceding sparse/dense test changing GEMM's
    # compiled parameters through process-global environment variables.
    config = {name: int(parameters.get(name, env.get(name, default)))
              for name, default in (("PARALLEL_MACS", 2), ("SPARSITY_EN", 1))}
    # Normalize explicit overrides and generated-header defaults so the dense
    # focused runner and the main matrix can share the same compiled image.
    kwargs["parameters"] = {**parameters, **config}
    env.update({name: str(value) for name, value in config.items()})
    kwargs["extra_env"] = env
    waves = os.environ.get("OPENEYE_CLUSTER_WAVES") == "1"
    kwargs["waves"] = waves
    kwargs["force_compile"] = False

    # Generate just the PE headers, without touching shared RTL timestamps.
    # Case-local copies also replace headers left by the former runner.
    for suffix in ("_PE", "_PE_cluster"):
        vh_file_creator.create_vh_file(SimpleNamespace(**config),
            str(case_dir / f"parameters{suffix}.vh"), toplevel="PE_cluster")
    headers = [case_dir / "parameters_PE.vh", case_dir / "parameters_PE_cluster.vh"]
    digest = hashlib.sha256()
    compile_options = {key: kwargs.get(key) for key in
                       ("toplevel", "defines", "parameters", "compile_args", "simulator")}
    compile_options.update(config=config, waves=waves)
    digest.update(json.dumps(compile_options, sort_keys=True).encode())
    # Include source contents and headers: Icarus's mtime check alone does not
    # track included files. A process-local cache isolates concurrent pytest runs.
    dependencies = [Path(path) for path in kwargs["verilog_sources"]]
    dependencies += sorted(Path(hdl_dir).rglob("*.vh")) + headers
    for path in dependencies:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    build_dir = Path(test_dir) / ".temp" / "pe_cluster_builds" / str(os.getpid()) / digest.hexdigest()[:20]
    build_dir.mkdir(parents=True, exist_ok=True)
    for header in headers:
        destination = build_dir / header.name
        if not destination.exists():
            destination.write_bytes(header.read_bytes())
    kwargs["includes"] = [str(build_dir), str(hdl_dir)]
    kwargs["sim_build"] = str(build_dir)
    kwargs["work_dir"] = str(case_dir)
    return cocotb_test.simulator.run(**kwargs)
