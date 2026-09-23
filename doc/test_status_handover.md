# OpenEye test status handover

Follow-up: section 7 records the convolution debugging milestone and supersedes
the original snapshot for the configurations explicitly retested there.

Snapshot date: 2026-09-19. Branch main, working tree clean, HEAD b273dab.
All results below were measured in one session on macOS (Icarus Verilog, cocotb, `openeye_env`). Every claim is tagged:

- OBSERVED: seen in a run this session (log evidence).
- FROM NOTES: taken from earlier project notes (2026-09-02 to 2026-09-18), not re-verified this session.
- ASSUMED: inference, not proven. Treat as a lead only.

## 1. Summary

| Area | Status |
|---|---|
| PE, PE_simple, data pipelines, iact stream constructor | Pass |
| PE_cluster dense and GEMM | Pass |
| FPGA GEMM / Dense (gemm_layer) | Pass, 26 of 26 |
| FPGA attention (host-orchestrated) | Pass, 10 of 10 |
| PE_cluster sparse with real zeros | Fails (about 4 percent of sampled cases) |
| FPGA convolution (any layer chained to a following layer) | Fails, 38 of 38 executed |
| cocotb_parallel system tests | Fail (stale testbench) |
| Net-level tests, wrapper, raw Verilog testbenches | Not run |

Main open problem: convolution on the FPGA top level. Main workaround for GEMM-like workloads: use the FPGA serial/DMA flow (`test/cocotb_fpga`).

## 2. How to run

- Activate: `source openeye_env/bin/activate`. Run pytest from inside the suite directory.
- macOS has no `timeout` command. Use background jobs and poll logs.
- Shell is zsh: unquoted `$var` is not word-split. Use `$(cat file)` or `${=var}` when passing many test ids.
- Test ids printed by `--collect-only` are rootdir-relative. Strip the `test/<suite>/` prefix when running from inside the suite directory.
- Use `-p no:cacheprovider -o log_cli=false -rfE` for quiet runs with a failure summary.
- Wall times measured (OBSERVED): gemm_layer 12m46s, attention 7m03s, conv_const 12m30s, single_layers 58m08s, conv3x3 sample of 8 tests 47m12s, PE_cluster sample of 162 tests 3m55s.
- Sim timeout in FPGA tests is 15 ms of sim time. A timed-out test costs 30 to 50 minutes wall.
- `test/cocotb_PE_cluster/test_PE_CLUSTER.py` now defaults to 32 smoke cases plus seven historical reproducers. `--cluster-regression=matrix` selects the 2,016-case seed-0 matrix plus six additional reproducers; `--cluster-regression=extended` restores all 32,256 cases.
- `test/cocotb_PE_cluster` now collects 64 tests by default, including focused sparsity, dense, and GEMM tests. See `test/cocotb_PE_cluster/README.md` for build reuse and opt-in waveforms.
- Constant-operand and ramp probes for value bugs (FROM NOTES): env vars `OPENEYE_CONST_IACTS`, `OPENEYE_CONST_WGHTS`, `OPENEYE_RAMP_IACTS` in `test/cocotb_fpga/OpenEye_FPGA_tb.py`. Vary the constant; one value cannot separate a data-independent DUT from a degenerate reference.

## 3. What works (OBSERVED)

| Suite (directory, file) | Result | What it covers |
|---|---|---|
| `test/`: `test_dense_reference.py`, `test_layer_adapter.py` | 13 passed | Python Dense reference math, layer adapter |
| `cocotb_PE` | 3 passed | Single PE, sparse and dense |
| `cocotb_PE_simple` | 16 passed | Simplified PE |
| `cocotb_iact_stream_constructor` | 6 passed | Split-K stream construction |
| `cocotb_data_pipeline` | 15 passed, 2 xfailed | Weight and iact SPAD encoding; xfails are the known all-zero-weight-row case |
| `cocotb_PE_cluster`: GEMM, dense | 11 passed (9 + 2) | Cluster GEMM approaches, dense conv |
| `cocotb_fpga/test_gemm_layer.py` | 26 passed | Row-stationary and output-stationary, PARALLEL_MACS 1 and 2, split-K, padding (K=31, 63), shapes 4x4 to 16x16 |
| `cocotb_fpga/test_attention.py` | 10 passed | 8 software scheduler tests plus 2 FPGA end-to-end tests (NUM_HEADS 1 and 2, SEQ_LEN 4, D_MODEL 8, output stationary) |

Caveat for the FPGA passes: geometry is small (CLUSTER_ROWS=2, K and N up to 32). Larger K, more cluster rows, and non-default NUM_GLB_* values are untested for GEMM.

## 4. What does not work

### 4.1 FPGA convolution: `compare_iact_storage` mismatch

Tests affected (all OBSERVED failing):

| Test | Failed / run | Failure point |
|---|---|---|
| `cocotb_fpga/test_conv_const.py` | 14 / 14 | `compare_iact_storage` assertion (SystemExit "FAILED 1 tests") |
| `cocotb_fpga/test_single_layers.py::test_single_conv_layer` | 16 / 16 | 14 at `compare_iact_storage`, 2 at SimTimeoutError |
| `cocotb_fpga/test_conv3x3_sparse.py` (sample of 8 of 96) | 8 / 8 | 7 at `compare_iact_storage`, 1 at SimTimeoutError |

Parametrization details (OBSERVED):
- single_layers ran CLUSTER_ROWS 8, 4, 2, 1, NUM_GLB_PSUM 4 and 2, NUM_GLB_WGHT 4 and 3. All 16 fail, so the failure does not depend on those knobs.
- The 2 timeouts are CLUSTER_ROWS=1 with NUM_GLB_PSUM=2 (both NUM_GLB_WGHT values). Other CLUSTER_ROWS=1 cases fail by mismatch instead.
- conv3x3 sample timeout: `[3-4-3-2-1-4-1-32-3-3-1-8]` (CLUSTER_ROWS=2, sparse, 32x1, 4 ch, 8 filters). Same test hung in earlier notes.
- conv_const covers CONST_VALUE 1, 2, 8, CLUSTER_ROWS 1 and 2, plus NUM_GLB_IACT/INPUT_CHANNELS variants (3,1) and (1,4).

Reasons, in order of evidence:

1. The interlayer psum to iact write-back ignores the computed psums (FROM NOTES, proven 2026-09-16). Constant-operand test: reference tracks the operand value (0x0, 0x0, 0x0101010101010101 for values 1, 2, 8) while the DUT always writes `0x2020202000000000`. So PE arithmetic is not the issue; the write-back data is independent of the input. Location suspected: `SEND_PSUM_TO_IACT` in `hdl/psum_pipeline.v`, rewritten in commit 904aa49 "Fixed Interlayer transmissions". Coordinate with the author before editing.
2. CLUSTER_ROWS=1 is a separate symptom (FROM NOTES): the iact buffer is never written (reads X) and the FSM stalls in RECEIVE_PSUMS_TO_IACT / PSUM_GET_RESULTS. Consistent with the CR=1 timeouts seen this session (OBSERVED timeouts, cause ASSUMED to be the same stall).
3. Iact double-buffer half-select is dropped (FROM NOTES): `iact_buffer_addr_reg` has the same width as `iact_buffer_addr`, so `{choose_iact_buffer, ...}` truncates and ping/pong aliases. Fix is a width change to `BRANCHES_WIDTH`. Would change behaviour; may expose a write-back that targets an unread half. `compare_iact_storage` addresses the buffer without a half offset today and needs the offset if this is fixed. Relation to the failures: ASSUMED secondary.
4. The check used to always return True and hid an empty buffer (FROM NOTES, 2026-09-03). Failures are therefore real, not new regressions. The Dense-layer stall class (fsm_cycle) was fixed 2026-09-04, so conv3x3 now mostly fails on values, not on hangs (OBSERVED: 7 of 8 mismatch, 1 of 8 hang).
5. The `test_conv3x3_sparse` hang (1 of 8) is pre-existing: stream sent, no results, WAIT_FOR_RESULTS (FROM NOTES, confirmed against baseline commit 3c9aacd). Root cause unknown.

### 4.2 PE_cluster sparse mode with real zeros

Tests affected (OBSERVED):
- `cocotb_PE_cluster/test_PE_CLUSTER_sparse.py::test_pe_cluster_weight_sparsity`: 3 failed of 14, parameters `[2-40]`, `[2-50]`, `[2-60]` (weight sparsity 40, 50, 60 percent). Failure: "Outcoming Partial Sums are not equal to Calculated data".
- `test_PE_CLUSTER.py::test_pe_cluster_conv`, sample of 162: 6 failed. Ids: `[1-1-13-40-10-6-3-2]`, `[1-2-0-50-20-6-3-4]`, `[1-2-1-40-0-6-2-3]`, `[1-2-5-60-30-10-3-2]`, `[1-2-9-60-20-10-2-3]`, `[1-2-10-50-0-8-3-2]`. Id field order: SPARSITY_EN, PARALLEL_MACS, SEED, SPARSE_WGHT, SPARSE_IACT, WGHTSIZE_X, IACTSIZE_Y, IACTSIZE_X. All 6 have SPARSITY_EN=1 and SPARSE_WGHT >= 40. Both PARALLEL_MACS values appear.

Reasons:
- Sparse-with-zeros verification gap (FROM NOTES). Four root causes were fixed 2026-09-05/06: skip count not carried into the psum address base in `PE.v`, missing `for` loop around read-after-write forwarding in `PE.v`, iact sender dropping zero activations in `pe_cluster_test_utils.send_to_iact_spad`, empty weight rows not encoded in `PE_cluster_tb.generate_spad`. The sampled failure rate went from 26 of 132 to 3 of about 129 in those notes.
- Remaining known gap (FROM NOTES): an entirely zero first weight row (`WGHT_ZERO_POS="0,1,2,3,4,5"`) and entirely zero rows generally. At high weight sparsity empty rows become likely, which matches the failing region (OBSERVED: all failures at 40 percent or more). Whether every failure this session is that one case is ASSUMED, not checked.
- The same notes list 8 PE-level sparse failures that occur only at PARALLEL_MACS=2 and link them to the weight-range shortfall (section 4.4). ASSUMED related.
- Reference tools: `doc/sparse_weight_encoding.md`, unit testbench `test/cocotb_data_pipeline/` (proves the weight pipeline emits a correct SPAD image, so the defect is in how PE.v consumes it or in the encoder).

### 4.3 cocotb_parallel

Tests affected (OBSERVED, ran `test_gemm_mode.py`, `test_single_layers.py`, `test_mnist.py`): 5 failed, 1 passed.

| Test | Error |
|---|---|
| `test_single_conv_layer[0-4-1-0-0-4-32-3-3-1-8]` | `TypeError: DRAMContents.__init__() missing 1 required positional argument: 'layer_parameters'` |
| `test_depthwise_conv_layer` (2 cases), `test_fc_layer[32-32]` | same TypeError (3 occurrences seen in log) |
| `test_single_pool_layer[10-7-7-7-1]` | `ValueError: No parameter file defined for toplevel ''` in `src/open_eye/vh_file_creator.py:180` |

Reason (FROM NOTES, matches OBSERVED errors): `OpenEye_Parallel_tb.py` predates the `src/open_eye` refactor. It still calls `DRAMContents(model)`, `write_initial_data_to_dram(model, ...)`, old `LayerParameters` and `make_ref` signatures. Current signatures are documented in the note. The serial branch of `rtl_test_utils.send_stream` also drives ports that no longer exist on `OpenEye_Parallel`. The fix is a port of the TB to the pattern in `test/cocotb_fpga/OpenEye_FPGA_tb.py`, not a mechanical patch. The single passing test is `test_gemm_mode.py::test_gemm_mode_plumbing[2]` (OBSERVED; `test_mnist.py` collects no tests). Earlier notes say it failed on a real assertion (`OS mode: cluster(0,0) PE(0,0) iact_sel_w=1, expected row index 0`); it passes now, so that was fixed since.

Recommendation: do not use cocotb_parallel for system checks. Use `test/cocotb_fpga`.

### 4.4 Known-open design issues that do not show in current passing tests (FROM NOTES)

These come from the 2026-09-18 notes. Today gemm_layer passes 26 of 26 (OBSERVED), which conflicts with the notes' "random data still fails" state. The likely explanation (ASSUMED, not verified) is the later commits in git log, for example "Fix Dense activation distribution across cluster rows" (2594391), after the notes were written. Re-check before trusting either.

- Dense/FC K-split across cluster rows: cluster row 1 received the same activations as row 0 instead of the second K slice. Question for the RTL author about how `current_x`, `x_pos_in_w_cycle`, and `iact_values_per_cluster_transmit` relate for FC. See 2594391 for the fix status.
- PARALLEL_MACS=2 weight shortfall: each PE received 3 weight ranges instead of 6. `write_wght_addr_storage` methods are dead code; something else fills the SPAD. Fix status unknown; gemm passes at PARALLEL_MACS=2 today (OBSERVED).
- Psum capture used column-major indexing while writers use row-major; fixed 2026-09-18 in `psum_pipeline.v` (FROM NOTES).

## 5. Not run this session

- `cocotb_fpga`: `test_nets.py`, `test.py`, `tflite.py`, `OpenEye_FPGA_gemmpass_probe_tb.py` (probe, not a test), 88 of 96 conv3x3 parametrizations.
- `cocotb_parallel`: `test_nets.py`, `test_mobilenet.py`.
- `cocotb_wrapper` (`OpenEye_Wrapper_tb.py`, own Makefile).
- Raw Verilog testbenches: `test/PE_dense_tb.v`, `PE_sparse_tb.v`, `PE_cluster_design/PE_cluster_tb.v`, `af_cluster/af_cluster_tb.v`, plus `test/Makefile`.
- `sw_fault_injection`, `simulation`, `multiplier`, router and mux/demux testbench directories.
- 32,094 of 32,256 `test_PE_CLUSTER.py::test_pe_cluster_conv` cases.
- Net-level tests are expected to fail on the conv write-back issue (ASSUMED) or on the stale parallel TB.

## 6. Suggested next steps

1. Debug the conv write-back in `SEND_PSUM_TO_IACT` using the constant-operand recipe on `test_conv_const.py` (about 1 minute per case at CLUSTER_ROWS=2). This unblocks single_layers, conv3x3 and the net tests.
2. Then decide on the iact double-buffer width fix (section 4.1, item 3).
3. Treat CLUSTER_ROWS=1 stalls separately (RECEIVE_PSUMS_TO_IACT / PSUM_GET_RESULTS).
4. For sparse PE_cluster, encode empty weight rows fully (xfail cases in `test/cocotb_data_pipeline/` will flip when fixed).
5. Port or retire `cocotb_parallel` tests.
6. Add a fast FPGA smoke test. Nothing under 7 minutes exists; conv paths cost up to an hour.
7. Add a sampled or pytest-marked subset of `test_PE_CLUSTER.py` so CI does not need 32k cases.

## 7. Convolution milestone follow-up (2026-09-19)

The first milestone is reached on a small dense configuration:
CLUSTER_ROWS=2, CLUSTER_COLUMNS=2, NUM_GLB_IACT=1, NUM_GLB_WGHT=3,
NUM_GLB_PSUM=4, BRANCHES=1, input 8x1 with 4 channels, 3x3 kernels,
and 4 output filters. This does not establish that convolution works for
all geometries or networks.

### Root causes established during this follow-up

- OBSERVED IN SOURCE: the legacy `LAYER=Convolution` creates two layers,
  despite the constant test's former single-layer description. Added an
  explicit `Convolution_Single` model and a testbench layer-count assertion;
  kept the legacy model unchanged.
- OBSERVED IN TRACE: bias DMA words hold two tightly packed 20-bit psums,
  but the RTL shifted each incoming word by 64 bits. Biases of 1 became
  `[1, 1, 0, 16, 16, 0, 256, ...]` across RAM lanes. Fixed payload width
  and the number of beats required to fill a psum RAM row.
- OBSERVED IN TRACE: convolution readout also shifted by 64 bits, dropping
  bits from successive 20-bit psums. The Python checker incorrectly expected
  three psums per beat. Aligned readout and reference with the existing
  two-psums-per-64-bit-beat schedule: tightly packed values, high bits zero.
- OBSERVED IN SOURCE AND REGRESSION: interlayer readout used `TRANS_WORDS`
  as its cluster stride and transpose width. With eight quantizers and four
  psum lanes per cluster it selected the wrong clusters and packed unused
  lanes. The stride now uses `NUM_GLB_PSUM`; selection and packing use only
  the active lanes of one cluster.

For the four-channel constant-8 case, the first convolution now produces
513 at the horizontal edges and 769 internally (including bias 1).
The default shift by 7 gives activation bytes 4 and 6, respectively.
The activation-buffer assertion and the second layer's final output
comparison both pass. Constants 1 and 8 are both checked; the ramp regressions
also distinguish spatial positions, with 20/32-bit accumulators and 4/8
quantizer instances.

### Validation results (OBSERVED)

| Check | Result | Coverage |
|---|---|---|
| True single-layer constant controls | 2 passed | Constants 1 and 8, 20-bit psums |
| Two-layer constant controls | 2 passed | Constants 1 and 8; intermediate buffer and final output |
| Two-layer activation ramp | 4 passed | 20/32-bit psums crossed with 4/8 quantizers |
| Two-layer activation ramp with distinct channel biases | 1 passed | Bias `1 + 128*f` makes channel permutations visible after quantization |
| FPGA GEMM K=8, N=8 | 4 passed | Both dataflows crossed with PARALLEL_MACS 1/2 |
| Python model/packing and existing Dense reference tests | 11 passed | Explicit model layer counts, signed DMA packing, Dense reference math |

The nine convolution passes were measured across focused runs during this
follow-up, not a run of the entire constant-convolution suite. The four ramp
cases plus the constant-1 two-layer case took 4m47s; channel-order validation
plus the four GEMM cases took 2m05s. No full-network or hardware-board run was
performed. The implementation is split into commits for the single-layer
control, DMA packing, interlayer write-back, and this handover.

### Scope and remaining issues

- OBSERVED: the original `(NUM_GLB_IACT=3, INPUT_CHANNELS=1)` constant-8
  reproducer still fails. After repairing bias packing, its PE output stream
  contains only the bias (1), before quantization. Its data-independent
  activation result is therefore not evidence that write-back alone ignores
  otherwise correct computed psums. Debug its activation/weight delivery next.
- CLUSTER_ROWS=1, BRANCHES>1, sparse inputs/weights, other NUM_GLB_PSUM values,
  larger images, and full networks were not revalidated in this milestone.
- Per-filter quantization coefficients, offsets, and saturation are not
  covered by these uniform-scale regressions.
- `TRACE_CONV_WRITEBACK=1` enables an optional cycle trace of psum capture,
  RAM selection, quantization, and activation writes.

Focused rerun (from the repository root):

```sh
source openeye_env/bin/activate
cd test/cocotb_fpga
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  'test_conv_const.py::test_conv_const_single_layer[2-1]' \
  'test_conv_const.py::test_conv_const_single_layer[2-8]' \
  'test_conv_const.py::test_conv_const_two_layers[iact1_ch4-2-1]' \
  'test_conv_const.py::test_conv_const_two_layers[iact1_ch4-2-8]' \
  test_conv_const.py::test_conv_ramp_writeback \
  test_conv_const.py::test_conv_channel_order \
  -q -p no:cacheprovider -o log_cli=false -rfE
```

Disabling pytest plugin autoload avoids an unrelated localhost socket opened
by pytest-rerunfailures. The convolution reference itself uses multiprocessing
IPC, so its simulations need permission to create local sockets.
