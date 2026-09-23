"""Selection guarantees and compile-cache isolation, without invoking RTL tools."""
from itertools import product

from cocotb_PE_cluster.conv_cases import (
    REPRODUCERS, SMOKE_WORKLOADS, convolution_cases,
)
from cocotb_PE_cluster import cluster_simulator


def test_cluster_selections_preserve_reproducers_and_full_matrix():
    smoke = convolution_cases("smoke")
    matrix = convolution_cases("matrix")
    extended = convolution_cases("extended")
    assert (len(smoke), len(matrix), len(extended)) == (39, 2022, 32256)
    assert set(REPRODUCERS) <= set(smoke) <= set(matrix) <= set(extended)
    expected = set(product((0, 1), (1, 2), range(16),
                           (0, 10, 20, 30, 40, 50, 60), (0, 10, 20, 30),
                           (10, 8, 6), (3, 2), (4, 3, 2)))
    assert set(extended) == expected


def test_smoke_exercises_each_workload_in_all_four_hardware_modes():
    smoke = convolution_cases("smoke")[:32]
    for mode in product((0, 1), (1, 2)):
        workloads = {(x, y, f, si, sw) for s, m, seed, sw, si, f, y, x in smoke
                     if (s, m) == mode}
        assert workloads == set(SMOKE_WORKLOADS)
    assert {case[-1] for case in smoke} == {2, 3, 4}
    assert {case[-2] for case in smoke} == {2, 3}
    assert {case[-3] for case in smoke} == {6, 8, 10}


def test_build_reuse_and_invalidation(tmp_path, monkeypatch):
    rtl = tmp_path / "hdl"
    rtl.mkdir()
    source = rtl / "PE_cluster.v"
    header = rtl / "included.vh"
    source.write_text("module PE_cluster; endmodule\n")
    header.write_text("// initial header\n")
    monkeypatch.setattr(cluster_simulator, "hdl_dir", str(rtl))
    monkeypatch.setattr(cluster_simulator, "test_dir", str(tmp_path))
    monkeypatch.delenv("OPENEYE_CLUSTER_WAVES", raising=False)
    # Environment leakage from other tests must not change this build.
    monkeypatch.setenv("PARALLEL_MACS", "1")
    monkeypatch.setenv("SPARSITY_EN", "0")
    calls = []
    monkeypatch.setattr(cluster_simulator.cocotb_test.simulator, "run",
                        lambda **kwargs: calls.append(kwargs))

    def run(case, **overrides):
        args = dict(sim_build=str(tmp_path / case), verilog_sources=[str(source)],
                    toplevel="PE_cluster", simulator="icarus", defines={"NO_TRACE": "TRUE"},
                    extra_env={"SEED": case}, force_compile=True, waves=True)
        args.update(overrides)
        cluster_simulator.run_cluster_simulation(**args)
        return calls[-1]

    first = run("seed0")
    second = run("seed1")
    assert first["sim_build"] == second["sim_build"]
    assert first["work_dir"] != second["work_dir"]
    assert second["extra_env"]["SEED"] == "seed1"
    assert second["extra_env"]["PARALLEL_MACS"] == "2"
    assert second["extra_env"]["SPARSITY_EN"] == "1"
    assert second["force_compile"] is False and second["waves"] is False
    explicit = run("explicit", parameters={"PARALLEL_MACS": 2, "SPARSITY_EN": 1})
    assert explicit["sim_build"] == first["sim_build"]

    header.write_text("// changed include without changing the top-level source\n")
    changed_header = run("header")
    assert changed_header["sim_build"] != first["sim_build"]
    source.write_text("module PE_cluster; wire added; endmodule\n")
    changed_source = run("source")
    assert changed_source["sim_build"] != changed_header["sim_build"]
    changed_mode = run("dense", parameters={"SPARSITY_EN": 0})
    assert changed_mode["sim_build"] != changed_source["sim_build"]
    changed_define = run("variant", defines={"USE_PE_SIMPLE": 1})
    assert changed_define["sim_build"] != changed_source["sim_build"]
    monkeypatch.setenv("OPENEYE_CLUSTER_WAVES", "1")
    with_waves = run("waves")
    assert with_waves["sim_build"] != changed_source["sim_build"]
    assert with_waves["waves"] is True
