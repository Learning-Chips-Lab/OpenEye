(output_stationary)=
# Output Stationary Dataflow (GEMM Mode)

Besides the default {ref}`row-stationary dataflow <dataflow>`, OpenEye can
run GEMM-shaped workloads (fully connected layers, matrix multiplications,
attention projections) with an **output-stationary (OS)** dataflow. The mode
is selected at runtime per layer; no recompilation of the hardware is
required.

## Core Principle

For `C = A x B (+ bias)` each PE keeps a tile of the output matrix `C`
stationary in its local psum SPad while the inner dimension `K` streams
through the array:

```
                 iact GLB bank 0   ->  PE row 0:  A[:, K0]
                 iact GLB bank 1   ->  PE row 1:  A[:, K1]
                 iact GLB bank 2   ->  PE row 2:  A[:, K2]

PE(row j, col i) accumulates:  C_tile(i) += A[:, Kj] x B[Kj, Ni]
```

- **iact routing** is fixed in hardware: with `gemm_mode = 1` every
  PE cluster binds iact GLB bank `j` to PE row `j`
  (`PE_cluster.gemm_mode_i` overrides the per-PE `iact_choose_i`).
- **weights** hold the matching `(K-slice x N-tile)` block of `B` per PE.
- **psums never move during compute**: each PE accumulates its output tile
  locally; the vertical psum drain runs once at the end and reduces the
  partial K-slices across the PE rows.

Compared to row-stationary, this maximizes psum reuse (zero intermediate
psum traffic) at the cost of iact reuse, which is the favourable trade-off
when there is no convolutional reuse to exploit — precisely the GEMM case.

## Hardware Support by Hierarchy Level

| Level | Mechanism |
|---|---|
| `PE.v` | Unchanged datapath: the psum SPad already accumulates locally (`CALCULATING` -> `WAIT_TO_SEND_PSUM` -> `SEND_PSUM`). The optional `SYSTOLIC_GEMM_EN` parameter additionally offers a systolic iact pass-through variant for `PE_simple`-based builds (Approach 3). |
| `PE_cluster.v` | `gemm_mode_i` input: when high, `iact_sel_w` of PE `(i, j)` is forced to row index `j`, turning the cluster into an output-stationary matrix multiplier. |
| `OpenEye_Cluster.v` | Forwards `gemm_mode_i` to the PE cluster; GLB and router paths are unchanged. |
| `OpenEye_Parallel.v` | `gemm_mode_i` configuration input, latched with `status_reg_enable_i` like `data_mode_i` and broadcast to all clusters. |
| `OpenEye_FPGA.v` | `gemm_mode` register in the DMA register map (`regmap.yaml` / `dma_storage.v`), driven per layer by the configuration words. |

## Software Support

- `OpenEyeParameters.DATAFLOW` (`"row_stationary"` default,
  `"output_stationary"`, overridable via the `DATAFLOW` environment
  variable) selects the dataflow globally; dense layers then run with
  `gemm_mode = 1`.
- `GemmMapper` (`gemm_mapper.py`) maps explicit GEMM layers; it reuses the
  dense stream layout (K split across cluster rows, PE rows and the per-PE
  iact SPad) and forces `gemm_mode = 1`.
- `LayerParameters.write_gemm_layer` handles layers named `gemm` / `matmul`
  and dense layers under the OS dataflow.
- Stream construction is described in {ref}`datastream_construction`.

## Usage

Row-stationary (default) and output-stationary runs differ only in the
configuration:

```bash
# Row-stationary dense layer (default)
pytest test/cocotb_fpga/test_gemm_layer.py -k row_stationary

# Same layer, output-stationary GEMM mode
pytest test/cocotb_fpga/test_gemm_layer.py -k output_stationary
```

or programmatically:

```python
os.environ["DATAFLOW"] = "output_stationary"
params = oep.get_oep(serial)          # params.DATAFLOW == "output_stationary"
# dense layers now map with gemm_mode = 1
```

## Verification

| Level | Test |
|---|---|
| PE / PE_cluster | `test/cocotb_PE_cluster/test_PE_CLUSTER_GEMM.py` — three GEMM approaches with numerical reference checks (Approach 1: software mapping only, Approach 2: `gemm_mode_i`, Approach 3: systolic pass-through with `PE_simple`). |
| OpenEye_Parallel | `test/cocotb_parallel/test_gemm_mode.py` — verifies latch semantics of `gemm_mode` and the row/bank binding inside every cluster of the hierarchy. |
| OpenEye_FPGA | `test/cocotb_fpga/test_gemm_layer.py` — full DMA flow of a GEMM layer under both dataflows against the TensorFlow reference. |

## Choosing a Dataflow

| Workload | Recommended dataflow |
|---|---|
| Convolutions (spatial reuse available) | row-stationary |
| Fully connected / GEMM / attention | output-stationary |
| Depthwise convolutions | row-stationary (dw mapper) |

The decision is per layer: a network can run its conv backbone
row-stationary and its classifier head output-stationary within the same
inference, since `gemm_mode` is part of the per-layer configuration words.
