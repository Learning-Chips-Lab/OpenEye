# OpenEye

| ![OpenEyeLogo](doc/figures/open_eye_logo.png) | **Open-Source Hardware Accelerator for Efficient Neural Network Inference** |
| - | - |

OpenEye is an open-source DNN inference accelerator originally developed at [FH Dortmund – University of Applied Sciences and Arts](https://www.fh-dortmund.de) and continued since March 2025 at the [University of Duisburg-Essen, Embedded Systems group](https://www.uni-due.de/ebs/).
It takes the ideas of [EyerissV2](https://arxiv.org/pdf/1807.07928) and makes them fully open, parameterizable, and FPGA-deployable — implementing a sparse, scalable systolic array for INT8 convolutions, depthwise convolutions, and fully-connected layers.

Where EyerissV2 is a fixed research chip, **OpenEye is designed to be resized at elaboration time**: cluster count, PEs per cluster, scratchpad depths, data widths, and router modes are all top-level parameters. This means you can target anything from a small edge FPGA to a large ASIC without touching a single line of RTL logic.

---

## What OpenEye can do

- **INT8 inference** of convolutional neural networks (CNNs), depthwise-separable convolutions, and fully-connected layers
- **Sparsity exploitation** at every level of the hierarchy: zero weights and zero activations are skipped, reducing power and latency
- **Row-Stationary Dataflow** for maximum data reuse and minimal off-chip bandwidth
- **On-chip 2×2 max-pooling** applied directly after the convolution
- **Fully parameterizable array** — cluster count, PEs per cluster, scratchpad depths, data widths, and router modes are all top-level Verilog parameters; no RTL changes needed to resize the accelerator
- **FPGA and ASIC targets** — `OpenEye_FPGA.v` wraps the ASIC compute core `OpenEye_Parallel.v` with DMA burst interfaces and block-RAM buffers for real FPGA deployment
- **Full end-to-end toolchain** — from a TFLite/Keras model through quantization, layer compilation, DMA stream generation, RTL simulation, and result verification, all from Python/CocoTB

---

## Key features at a glance

| Feature | Detail |
|---|---|
| Data types | INT8 activations & weights, 20-bit psum accumulator |
| Default array size | 2 × 8 clusters, 12 PEs/cluster → 192 PEs |
| **Array scalability** | `CLUSTER_COLUMNS`, `CLUSTER_ROWS`, `NUM_GLB_PSUM`, `NUM_GLB_WGHT` are all top-level parameters — change them without touching RTL logic |
| Dataflow | Row-Stationary |
| Sparsity | Structural sparsity on both iact and wght |
| On-chip pooling | 2×2 max-pooling |
| Host interface | 64-bit DMA AXI-stream-style handshake |
| Quantization | Per-filter affine: `q = (mant × (psum + offset)) >>> exp` |
| HDL | Verilog (SystemVerilog-compatible subset) |
| License | Solderpad Hardware License v2.1 (SHL-2.1) |

### Scalability vs. EyerissV2

EyerissV2 introduced the concept of a hierarchical, reconfigurable PE array — but as a fabricated ASIC its dimensions are fixed.
OpenEye exposes the same architectural knobs as compile-time Verilog parameters:

| Parameter | What it controls | Example values |
|---|---|---|
| `CLUSTER_COLUMNS` | Width of the cluster grid | 1, 2, 4 … |
| `CLUSTER_ROWS` | Height of the cluster grid | 2, 4, 8 … |
| `NUM_GLB_WGHT` | PE rows per cluster (= weight GLBs) | 3, 4, 6 … |
| `NUM_GLB_PSUM` | PE columns per cluster (= psum GLBs) | 4, 8 … |
| `IACT_PER_PE` / `WGHT_PER_PE` / `PSUM_PER_PE` | Scratchpad depths | tune to fit target BRAM |
| `SERIAL` / `PARALLEL_MACS` | Serial vs. parallel MAC inside each PE | 1 / 2, 4 … |
| `SPARSITY_EN` | Enable or disable sparse encoding | 0 or 1 |

Because all routing, buffering, address generation, and DMA logic is derived from these parameters via `generate` blocks and `localparam` arithmetic, you get a correctly wired accelerator of any size after a single re-elaboration — no manual wiring, no RTL edits.

---

## Repository layout

```
OpenEye/
├── hdl/                        RTL source files
│   ├── OpenEye_FPGA.v          Top-level FPGA wrapper (~3050 lines)
│   ├── OpenEye_Parallel.v      ASIC compute core
│   ├── OpenEye_Cluster.v       Top-level cluster
│   ├── PE_cluster.v            One cluster of PEs
│   ├── PE.v                    Processing element (MAC unit)
│   ├── iact_stream_constructor.v  Converts raw pixels → sparse iact streams
│   ├── dma_storage.v           Auto-generated register-map decoder
│   ├── varlenFIFO.v            Variable-length FIFO
│   ├── RST_SYNC.v              2-FF reset synchronizer
│   ├── RAM_SP.v / RAM_DP.v     SRAM primitives
│   ├── router_iact/wght/psum.v  Flexible routing network
│   └── include/
│       └── regmap_params.vh    Parameter header for dma_storage
├── src/                        Python host-side toolchain
│   └── open_eye/               Package: model loader, layer adapter,
│                               DMA stream generator, config compiler
├── test/                       Testbenches (see below)
│   ├── cocotb_PE/              Unit test for a single PE
│   ├── cocotb_PE_cluster/      Unit test for a PE cluster
│   ├── cocotb_parallel/        Integration tests for OpenEye_Parallel
│   ├── cocotb_fpga/            System tests for OpenEye_FPGA
│   └── simulation/             Pre-computed reference datasets
│       └── mnist_quantized_model/
├── doc/
│   ├── OpenEye_FPGA_documentation.md  Detailed module documentation
│   ├── demo_notebooks/
│   │   └── simple_cnn_demo.ipynb   Quickstart tutorial (start here!)
│   └── figures/
└── requirements.txt
```

---

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Required tools:
- **Python 3.10+**
- **Icarus Verilog** (`iverilog`) — default simulator; `brew install icarus-verilog` on macOS
- **CocoTB** — installed via pip (see `requirements.txt`)
- Optional: **Verilator ≥ 5.020** (faster simulation; see note below)
- Optional: **GTKWave** for waveform inspection

> **Verilator patch** — older versions require a one-line fix in `include/verilatedos.h`:
> ```diff
> - #define VL_VALUE_STRING_MAX_WORDS 64
> + #define VL_VALUE_STRING_MAX_WORDS 128
> ```

### 2. Try the demo notebook

The fastest way to understand the full workflow is the Jupyter notebook:

```
doc/demo_notebooks/simple_cnn_demo.ipynb
```

It walks through:
1. Building a minimal CNN (Conv2D + Dense) on MNIST
2. INT8 quantization with TFLite
3. Loading the model with the OpenEye Python toolchain
4. Inspecting the hardware mapping (ops, PE utilization, cycle estimate)

### 3. Run a single-layer simulation

```bash
cd test/cocotb_fpga
make run
```

This simulates one convolutional layer (28×28 input, 32 filters, 3×3 kernel, stride 1) through the full `OpenEye_FPGA` top-level. Parameters can be changed directly in the [Makefile](test/cocotb_fpga/Makefile):

```makefile
export LAYER=Convolution      # or FC, Depthwise_Convolution, Pooling
export NUM_FILTERS=32
export KERNEL_SIZE=3
export INPUT_SIZE_X=28
export INPUT_SIZE_Y=28
export STRIDE=1
export INPUT_CHANNELS=1
```

Simulation dumps an FST waveform to `sim_build/` which can be opened with GTKWave:

```bash
gtkwave sim_build/OpenEye_FPGA.fst
```

---

## Testing strategy

OpenEye uses a layered test hierarchy with [CocoTB](https://www.cocotb.org/) and [pytest](https://docs.pytest.org/):

| Level | Location | What it tests |
|---|---|---|
| **PE unit** | `test/cocotb_PE/` | Single processing element: MAC correctness, sparse activation handling, scratchpad addressing |
| **PE cluster** | `test/cocotb_PE_cluster/` | Cluster of PEs: router configuration, intra-cluster data reuse, psum accumulation |
| **OpenEye_Parallel** | `test/cocotb_parallel/` | ASIC compute core: single layers, full MNIST CNN, MobileNet depthwise layers |
| **OpenEye_FPGA** | `test/cocotb_fpga/` | Full FPGA system: DMA loading, iact conversion, quantization, max-pooling, layer-to-layer routing |

### Running the tests

**PE unit test:**
```bash
cd test/cocotb_PE
make run
```

**Full FPGA system test (single layer):**
```bash
cd test/cocotb_fpga
make run
```

**Integration test suite (pytest):**
```bash
cd test/cocotb_parallel
pytest
```

**Full network tests** (AlexNet, ResNet, MobileNet) are also available in `test/cocotb_parallel/test_nets.py` and `test/cocotb_parallel/test_mobilenet.py`.

All tests generate FST waveforms by default (controlled by `IVERILOG_DUMPER=fst`). The notebook `test/cocotb_PE/analysis.ipynb` can be used for post-simulation waveform analysis.

---

## Architecture overview

```
Host CPU  ──DMA (64-bit)──►  OpenEye_FPGA
                              │
                              ├─ RST_SYNC          (metastability-safe reset)
                              ├─ dma_storage        (config register file)
                              ├─ RAM_SP ×32         (iact double-buffer, ping-pong)
                              ├─ iact_stream_constructor ×(CC×CR)
                              │    └── encodes raw pixels → sparse iact streams
                              ├─ RAM_SP             (weight staging buffer)
                              ├─ RAM_SP ×(CC×CR×P/2) (psum staging buffers)
                              └─ OpenEye_Parallel   (ASIC compute core)
                                   └─ OpenEye_Cluster ×(CC×CR)
                                        └─ PE_cluster
                                             └─ PE ×(WGHT×PSUM)
```

The main FSM sequences through:
`IDLE → GET_PARAMETERS → GET_ROUTER_CONFIG → GET_IACT → GET_WGHT → GET_BIAS → GET_QUANTIZE → GET_OFFSET → START_CONVERTER → CONVERT_IACT → WAIT_FOR_RESULTS`

and the PSUM FSM handles result collection and optional layer chaining:
`PSUM_IDLE → CALCULATE_PSUM → PSUM_GET_RESULTS → PSUM_SEND_RESULTS`

Full FSM documentation, all parameters, port descriptions, and key data-path explanations are in [doc/OpenEye_FPGA_documentation.md](doc/OpenEye_FPGA_documentation.md).

---

## Documentation

| Resource | Description |
|---|---|
| [doc/OpenEye_FPGA_documentation.md](doc/OpenEye_FPGA_documentation.md) | Detailed technical documentation of the top-level FPGA module: FSMs, parameters, ports, submodule descriptions, data paths |
| [doc/demo_notebooks/simple_cnn_demo.ipynb](doc/demo_notebooks/simple_cnn_demo.ipynb) | Quickstart notebook — minimal CNN, quantization, hardware mapping |
| `doc/build/html/` | Sphinx HTML documentation (build with `make html` in `doc/`) |

To build the HTML docs locally:

```bash
cd doc
pip install -r requirements.txt
make html
# open build/html/index.html
```

---

## Licensing

OpenEye is covered by the Solderpad Hardware License, Version 2.1 (see [LICENSE](LICENSE) for full text).
