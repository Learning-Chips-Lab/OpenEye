# Deployment Flow: ONNX to FPGA Execution

This chapter describes the complete flow from a trained ONNX model to
inference on an FPGA board, using the model artifact toolchain
(`src/open_eye/artifact.py`) and the portable C runtime
(`fpga/sw/runtime/`).

## Overview

All model preparation happens offline on a development machine. The
target board only replays precompiled DMA streams; it never parses ONNX
and needs no firmware rebuild when the model changes.

```
 Development machine (Python)                     Target board (C runtime)
 ────────────────────────────                     ────────────────────────
 ONNX model
   │  onnxruntime: graph opts +
   │  static quantization (QOperator)
   ▼
 quantized ONNX
   │  open_eye.model_loader /
   │  onnx2model.py
   ▼
 OpenEye layer representation
   │  layer_parameters.py (tiling,
   │  repetitions) + layer mappers
   ▼
 per-transfer DMA word streams
   │  open_eye.artifact
   ▼                                                artifact in DDR
 model.oeye  ──── SD card / network / JTAG ────►      │ openeye_model_parse()
                                                      ▼
                                                    openeye_run()
                                                      │ HAL: AXI DMA
                                                      ▼
                                                    OpenEye_FPGA (PL)
```

The accelerator has no instruction set in the conventional sense: the
"executable" is the sequence of 64-bit words pushed into its DMA stream
interface (see {ref}`fpga`). Compiling a model for OpenEye therefore
means generating these streams; running it means replaying them in
order and reading the result stream back.

## Stage 1: Model preparation (ONNX)

Starting point is a quantized INT8 ONNX model. Models exported from
PyTorch, Keras or TFLite can be used as well via the unified loader
(`open_eye.model_loader.load_model`); this chapter follows the ONNX
path.

Recommended preprocessing with ONNX Runtime before import:

- run the ORT graph optimizer or `onnx-simplifier` (constant folding,
  BatchNorm folding, Conv+Relu fusion, shape inference),
- quantize with `onnxruntime.quantization.quantize_static(...,
  quant_format=QuantFormat.QOperator)`.

The QOperator format emits exactly the operator set the importer
understands: `QLinearConv`, `QLinearMatMul`, `MaxPool`. The importer
(`open_eye/onnx2model.py`) also accepts float `Conv`, `Gemm`, `Relu`
and `BatchNormalization` nodes, but preprocessing keeps its job small
and deterministic.

## Stage 2: Import and mapping (Python, offline)

`create_model_from_onnx` converts the ONNX graph into the internal
layer representation (`ONNXConv2d`, `ONNXLinear`, ...), which the
adapter layer (`open_eye/layer_adapter.py`) exposes through the
interface expected by `LayerParameters`.

`LayerParameters` computes, per layer, how the computation is tiled
onto the cluster array: how many transmissions ("repetitions") are
needed, which PEs and routers are used, and the refresh schedule. The
layer mappers (`conv_mapper.py`, `dense_mapper.py`, `dw_mapper.py`,
`pooling_mapper.py`) then generate the actual DMA word streams.

Each transfer (one layer repetition) is the concatenation of six
sections, in the order the hardware consumes them:

```
status | iact | wght | psum | quantize | offset
```

`status` carries the working parameters decoded by `dma_storage`;
`iact` carries raw activation pixels, which the wrapper converts to
sparse streams on-chip (`iact_stream_constructor`, `CONVERT_IACT`
state); the remaining sections carry weights, bias/psum data and the
post-processing parameters.

Not every layer produces a readback. The wrapper chains layers on-chip
where possible: quantized psums are written back into the URAM iact
double-buffer as next-layer activations (`RECEIVE_PSUMS_TO_IACT`), and
2x2 max-pooling runs inside the wrapper (`MAXPOOLING_READ/SEND`). Only
the final layer (or an explicit spill) crosses the DMA boundary
outward.

## Stage 3: Packaging (.oeye artifact)

`open_eye.artifact` packs the streams into a single deployable file.

From an existing stream dump directory (as written by the cocotb test
infrastructure into `test/cocotb_fpga/demo/`):

```console
$ python -m open_eye.artifact build test/cocotb_fpga/demo -o model.oeye
$ python -m open_eye.artifact info model.oeye
```

Or directly from mapper output, which additionally records the
position of the iact section so the runtime can patch new input data
into the first transfer at inference time:

```python
from open_eye.artifact import ArtifactBuilder, hw_param_hash

b = ArtifactBuilder(hw_hash=hw_param_hash(params))
b.add_layer_from_stream(mapper.get_stream(), layer_id=0,
                        patchable_input=True)
...
b.set_reference(expected_output_bytes)   # optional on-target self-test
b.write("model.oeye")
```

The artifact contains:

- one DMA blob per transfer, byte-exact as sent over the 64-bit AXI
  stream (little-endian words),
- per-transfer metadata: length, expected readback bytes, optional
  input-patch window,
- model I/O metadata: input/output sizes and quantization parameters,
- a hash of the hardware parameterization, so an artifact built for a
  different cluster configuration is rejected at load time,
- optionally the reference output for a self-test on the board.

The binary layout is documented in the module docstring of
`src/open_eye/artifact.py` and mirrored by `openeye_runtime.h`.

## Stage 4: Host-side verification (no hardware)

The runtime builds and tests on any POSIX machine using a mock HAL that
serves readbacks from the embedded reference:

```console
$ cd fpga/sw/runtime
$ make check ARTIFACT=path/to/model.oeye
```

This validates the artifact parses, all transfers are well-formed, the
hardware-hash check works, and the self-test detects corrupted output.

For functional verification of the streams themselves, the cocotb
testbench (`test/cocotb_fpga/OpenEye_FPGA_tb.py`) drives the same
blobs into the simulated `OpenEye_FPGA` and compares against
`dma_stream_ref.txt`. Simulation and board therefore consume identical
bytes.

## Stage 5: Execution on the board

### Hardware side

The Vivado block design instantiates `fpga/hdl/open_eye_axi_v1_0.v`
(the AXI wrapper around `OpenEye_FPGA`) together with a Xilinx AXI DMA
in simple mode: MM2S feeds the 64-bit input stream, S2MM receives the
result stream.

### Software side

The runtime consists of three small pieces:

| File | Role |
|---|---|
| `openeye_runtime.c/.h` | artifact parser + transfer replay (freestanding C99) |
| `openeye_hal.h` | 4-function platform interface |
| `hal_xaxidma.c` | bare-metal Zynq backend (XAxiDma, polling) |
| `hal_mock.c` | host-side test backend |

The HAL contract has one hardware-specific subtlety: the accelerator
begins emitting results while the final transfer is still being
written, so the readback must be armed before the write
(`dma_read_start` / `dma_write` / `dma_read_wait`).

A minimal bare-metal application (`main_baremetal.c`):

```c
openeye_model model;
openeye_model_parse(artifact_in_ddr, artifact_len, expected_hash, &model);

openeye_hal hal;
hal_xaxidma_init(&hal, XPAR_AXIDMA_0_DEVICE_ID);

/* self-test against the embedded reference ... */
openeye_selftest(&model, &hal, output, model.output_len);

/* ... or real inference with a new input image */
openeye_run(&model, &hal, input_pixels, input_len,
            output, model.output_len);
```

The artifact is loaded to DDR by whatever the boot flow provides
(XSCT/JTAG `dow -data`, U-Boot `fatload`, or a filesystem read under
Linux). Deploying a new model means copying a new `.oeye` file; the
firmware is model-agnostic.

On Zynq Linux the same runtime links against a HAL backend using
`udmabuf`/UIO or the dmaengine API instead of XAxiDma; on a RISC-V SoC
against the platform DMA driver. Only the four HAL functions change.

### Input and output data

Input: raw quantized pixels, laid out exactly as the iact section of
the first transfer expects them (the wrapper performs the sparse
encoding on-chip). Quantize with the artifact's `input_scale` /
`input_zero_point`.

Output: the final readback delivers packed psum words; dequantize with
`output_scale` / `output_zero_point`.

## Current limitations

- Runtime input patching (`openeye_run` with a non-NULL input) requires
  artifacts built with `add_layer_from_stream(...,
  patchable_input=True)`; artifacts built from text dumps via
  `from_demo_dir` carry no patch window and run with baked-in
  activations only. The byte layout of runtime-injected input inside
  the iact section has not yet been validated on hardware.
- Readback scheduling is derived from the reference dump (final
  transfer only); models that spill intermediate psums to DDR need the
  `out_bytes` fields set explicitly when building the artifact.
- The legacy flow via `fpga/sw/generate_dma_header.py` (streams baked
  into C headers) remains available but is superseded by the artifact
  runtime.
