(datastream_construction)=
# Datastream Construction

This chapter documents how the Python toolchain (`src/open_eye`) turns a
neural-network layer into the data streams that the OpenEye hardware
consumes: which modules participate, what each stream contains, how the
words are packed bit by bit, and how the layout differs between the
row-stationary and the output-stationary dataflow.

## Pipeline Overview

```
model (Keras / TFLite / ONNX)
        |
        v
LayerParameters (layer_parameters.py)     <- mapping decisions per layer
        |
        v
LayerMapper subclass (conv/dw/dense/pooling/gemm_mapper.py)
   |         |          |
   v         v          v
IactStream WghtStream PsumStream          <- per-datatype stream creators
   \         |          /
    \        |         /
     +-------+--------+
             |
             v
   stream[channel] lists                  <- consumed by the testbenches
             |                               or packed into DMA words
             v
   OpenEye_FPGA DMA port (serial) or
   OpenEye_Parallel ports (parallel)
```

Participating modules:

| Module | Responsibility |
|---|---|
| `layer_parameters.py` | Computes the per-layer mapping: how many PEs, clusters, SPad words, transmissions and refresh cycles a layer needs. Sets `fully_connected`, `gemm_mode`, skip flags. |
| `layer_mapper.py` | `LayerMapper` base class; orchestrates the three stream creators and assembles the final stream structure. |
| `conv_mapper.py` / `dw_mapper.py` | Row-stationary convolution / depthwise mappers. |
| `dense_mapper.py` | Fully-connected mapper (K split across banks, output stationary in the PE psum SPad). |
| `gemm_mapper.py` | Output-stationary GEMM mapper; dense layout plus `gemm_mode = 1`. |
| `pooling_mapper.py` | Pooling layer configuration (uses the conv datapath in pooling mode). |
| `iact_stream_mapper.py` | Input-activation stream creators (`Conv`, `Dense`, `Dw` variants). |
| `wght_stream_mapper.py` | Weight stream creators; produce the two-level (address + data) SPad images. |
| `psum_stream_mapper.py` | Bias / partial-sum stream creators. |
| `generator.py` | Generates `dma_storage.v`, `regmap_params.vh` and `regmap_pack.py` from `regmap.yaml`. |
| `regmap_pack.py` | Generated; packs/unpacks the layer-configuration registers into 64-bit DMA words. |
| `stream_dicts.py` | Index dictionaries that name the channels of the assembled stream. |

## Stream Channels

`LayerMapper.make_stream()` fills one list per channel
(see `stream_dicts.py`):

- **Serial (DMA / OpenEye_FPGA)** — `stream_serial_dict`:
  `status`, `router_iact`, `router_wght`, `router_psum`,
  `iact_data`, `wght_data`, `psum_data`.
  In practice the mappers concatenate the router words onto the `status`
  channel, so the FPGA receives one contiguous configuration block.
- **Parallel (direct ports / OpenEye_Parallel)** — `stream_parallel_dict`:
  `status`, `iact`, `wght`, `psum`, `quantize`, `offset`.
  The `status` channel is itself a list indexed by `status_dict`
  (`data_mode`, `used_X_cluster`, ..., `gemm_mode`), and
  `rtl_test_utils.send_stream()` drives each entry onto the matching
  OpenEye_Parallel input port.

## Serial DMA Stream Layout (OpenEye_FPGA)

The FPGA consumes one 64-bit word per DMA beat. Per layer transmission the
words arrive in the order of the main FSM states:

1. **Configuration words** (`GET_PARAMETERS`): the register words produced
   by `regmap_pack.pack_registers()`. `regmap.yaml` is the single source of
   truth for this layout; the generator assigns each register a
   transmission index and bit position (first-fit into 64-bit words) and
   emits the identical layout to hardware (`dma_storage.v`) and software
   (`regmap_pack.py`). The current map needs six words and includes among
   others `stride_x/y`, `kernel_size_x/y`, `iact_size_x/y`, `filters`,
   `fully_connected_layer`, `pooling_mode` and `gemm_mode` (the dataflow
   select bit added for the output-stationary mode).
2. **PE enable bitmap** (`GET_PARAMETERS`, remaining words): one bit per PE,
   linearized as `cluster_x, cluster_y, pe_y, pe_x` and split into 64-bit
   segments; taken from `layer_params.computing_mx`.
3. **Router configuration** (`GET_ROUTER_CONFIG`): three word groups, one
   per NoC:
   - iact routers: 6 bits per router, `floor(64/6) = 10` routers per word,
   - wght routers: 1 bit per router, 64 per word,
   - psum routers: 3 bits per router, 21 per word.
   The values encode source/pass/accumulate behaviour per GLB router
   (see `write_router_iact/wght/psum` in the mappers).
4. **Input activations** (`GET_IACT`): raw pixel values in two's complement,
   eight 8-bit values per 64-bit word. The hardware
   `iact_stream_constructor` converts them on the fly into the sparse
   (value + overhead) SPad streams and generates the per-PE `iact_choose`
   selectors. For dense/GEMM layers the flattened input vector is split
   into K-slices so that GLB bank `j` receives slice `j`
   (`DenseIactStreamMapper.get_iact_stream`).
5. **Weights** (`GET_WGHT`): the pre-encoded two-level SPad image produced
   by `wght_stream_mapper`: first the address SPad words (column pointers),
   then the data SPad words (weight payload plus `ignore_zeros` sparsity
   offsets, `PARALLEL_MACS` values per entry).
6. **Bias / partial sums** (`GET_BIAS`): initial psum values, one
   `DATA_PSUM_BITWIDTH` value per cluster column packed per word
   (`write_psum_data_glb`).
7. **Quantization** (`GET_QUANTIZE`): 512 words; each word carries two
   `(mantissa, exponent)` pairs at bit offsets 0/25 and 32/57. Applied as
   `q = (mant * (psum + offset)) >>> exp`.
8. **Offsets** (`GET_OFFSET`): 128 words with eight 8-bit offsets each.

The skip flags (`skipIact_reg`, `skipWght_reg`, `skipPsum_reg`) suppress
individual phases when data from the previous transmission can be reused.

## Parallel Stream Layout (OpenEye_Parallel)

In parallel mode the same logical content is delivered without DMA packing:

- `status` entries are driven directly onto configuration ports
  (`stride_x_i`, `filters_i`, `gemm_mode_i`, ...) while
  `status_reg_enable_i` is high; router modes are packed into the
  `router_mode_*_i` vectors.
- `iact`, `wght` and `psum` are nested lists indexed
  `[cluster_x][cluster_y][router]`, each containing the ready-encoded SPad
  words for that GLB bank. Because no hardware converter exists on this
  path, the Python side performs the sparse encoding itself
  (`set_sparse_stream`).

## Sparse Encoding

Both iacts and weights use a compressed format inside the PE SPads:

- **Iact**: address SPad holds per-column counts, data SPad holds
  `(value, overhead)` pairs where `overhead` encodes the number of skipped
  zeros. Dense mode (`SPARSITY_EN=0`) stores plain values.
- **Wght**: address SPad holds base pointers per iact index; data SPad
  entries pack `PARALLEL_MACS` weights, each with an `ignore_zeros` field
  used as psum-address offset for zero-skipping.

## Row-Stationary vs. Output-Stationary Layout

The two dataflows share the stream channels; they differ in *what* is
placed *where*:

| Aspect | Row-stationary (conv) | Output-stationary (GEMM) |
|---|---|---|
| Mapper | `ConvMapper` / `DWMapper` | `GemmMapper` (or `DenseMapper` with `DATAFLOW=output_stationary`) |
| iact per GLB bank | image rows, diagonally multicast; PE selects bank via `iact_choose` | K-slice `j` in bank `j`; binding fixed in hardware (`gemm_mode=1`) |
| wght per PE row | one filter row per PE | `(K-slice x N-tile)` block of B per PE |
| psum | accumulated along the PE column (vertical drain) | output tile stationary in the PE psum SPad; drained once at the end |
| Config difference | `gemm_mode=0` | `gemm_mode=1` (DMA register / `gemm_mode_i` port) |

See {ref}`dataflow` for the row-stationary background and
{ref}`output_stationary` for the GEMM mode details.

## Extending the Register Map

To add a configuration field, append it to `regmap.yaml`
(`test/cocotb_fpga/regmap.yaml`, kept in sync with
`hdl/config/regmap.yaml`) and regenerate:

```bash
cd test/cocotb_fpga
python -m open_eye.generator . ../../hdl/include ../../hdl
```

This rewrites `hdl/include/regmap_params.vh`, `hdl/dma_storage.v` and
`src/open_eye/regmap_pack.py` in one step, guaranteeing hardware and
software agree on the bit layout. `pack_registers()` defaults missing
values to 0, so existing mappers keep working when new registers are
appended. The FPGA testbenches additionally regenerate these files into
their `sim_build` directory with the environment of the concrete test run.
