(attention)=
# Attention on OpenEye (Phase 0: Host-Orchestrated)

OpenEye can execute single- and multi-head attention by decomposing the
operation into GEMM passes that run on the {ref}`output-stationary datapath
<output_stationary>`, with softmax and requantization performed on the host
between passes. This chapter documents the Phase-0 flow implemented in
`src/open_eye/attention_scheduler.py` and the roadmap towards fully
on-chip attention.

## Decomposition

For input `X` (`seq_len x d_model`, INT8) and INT8 projection weights:

```
stage 1  projection   Q = X W_Q,  K = X W_K,  V = X W_V     on-chip
         (host: requantize Q, K, V to INT8, power-of-2 shifts)
stage 2  score        S_h = Q_h K_h^T                        on-chip
         (host: softmax(S_h * scale / sqrt(d_k)), quantize -> P_h)
stage 3  context      O_h = P_h V_h                          on-chip
         (host: concat heads, requantize -> INT8)
stage 4  output       out = O W_O                            on-chip
```

Each on-chip pass is one GEMV per token through the dense/GEMM datapath
(`GemmMapper`, `gemm_mode = 1`): the weight operand stays resident in the
PE array, the token vector streams in as iacts, and the raw integer psums
stream back over DMA. The total is `(3 + 2*heads + 1) * seq_len` passes.

**No transpose hardware is needed for** `Q K^T`: the DRAM weight layout is
`weights[out][in]`, so using the K matrix itself as the weight operand
(`out` = token j, `in` = channel c) directly computes
`S[i][j] = sum_c Q[i][c] * K[j][c]`.

**Multi-head** costs no additional hardware: heads are column slices of
the projection results; stages 2 and 3 simply run per head, and the head
concatenation before `W_O` happens in the host requantization step.

## Fixed-Point Contract

Phase 0 pins down the exact arithmetic that later hardware phases must
reproduce:

- On-chip passes return **exact integer accumulations** (`W @ x`, biases
  zero on the dense path; the 20-bit psum range bounds
  `inner_dim * |a|max * |w|max`, which the scheduler respects by
  requantizing operands to INT8 between stages).
- **Weight operands must be zero-free.** The serial dense weight stream is
  zero-compressed and the datapath rejects zero weight entries (verified
  with `OpenEye_FPGA_gemmpass_probe_tb`: zero *iacts* are tolerated, zero
  *weights* corrupt or hang the pass; the DRAM writer applies the same
  +-1 replacement to random dense weights). The scheduler therefore
  produces zero-free K, V and P tensors (`requantize_pow2(..., nonzero=True)`,
  softmax probabilities clipped to `[1, 127]`) and sanitizes the static
  projection weights the same way — a one-LSB perturbation that is part of
  the deterministic contract.
- Host requantization uses power-of-two shifts with
  round-half-away-from-zero (`requantize_pow2`), deterministic on
  integers.
- Softmax runs in float on the host, on the dequantized scores scaled by
  `1/sqrt(d_k)`; probabilities are quantized to `[0, 127]`.
- Scales are tracked end to end, so the final integer output dequantizes
  to a value comparable against the float attention reference.

Because every step is deterministic, the hardware execution can be checked
**bit-exactly** at three levels: per pass (`GemmPass.expected_int()`),
against the host-executed schedule (`AttentionScheduler.run_on_host()`),
and — with tolerance — against the float reference
(`float_reference()`).

## Usage

```python
from open_eye.attention_scheduler import AttentionScheduler

sched = AttentionScheduler(x, w_q, w_k, w_v, w_o, num_heads=2,
                           s_x=1/64, s_w=1/64)
for stage in sched.stages():
    for p in stage.passes:
        p.result = execute_gemm_pass(p)   # on-chip, or p.expected_int()
    stage.finalize()                       # host softmax / requantization
out_int, out_scale = sched.final_output()
```

The executor receives, per pass, an INT8 weight matrix (`p.weight`,
`[N, K]` in DRAM layout), an INT8 input vector (`p.iact`) and must return
the N raw psums. The cocotb testbench
`test/cocotb_fpga/OpenEye_FPGA_attention_tb.py` implements this executor
on top of the full DMA flow.

## Verification

| Test | Level | Runtime |
|---|---|---|
| `test/cocotb_fpga/test_attention.py::test_attention_scheduler_software` | Host integer model vs. float reference; INT8 range and zero-free weight checks; determinism; 1/2/4 heads up to seq 16, d_model 32 | < 1 s |
| `test/cocotb_fpga/test_attention.py::test_attention_fpga` | Full attention through OpenEye_FPGA (seq 4, d_model 8, 1 and 2 heads = 24/32 DMA passes); every pass and the final result bit-exact vs. the host model, ~1% error vs. the float reference | ~1.5 min per case |
| `test/cocotb_fpga/OpenEye_FPGA_gemmpass_probe_tb.py` | Operand-constraint probes for the dense datapath (control / zero-iact / zero-weight) | ~1 min each |

## Limitations of Phase 0 and Roadmap

Phase 0 is a functional reference, not a fast implementation: every
dynamic operand (K, P, V) makes a host round-trip, and each pass pays the
full per-layer DMA overhead (config, quantize and offset streams).
Additionally, the dense datapath has a minimum problem size (8x8 GEMVs do
not complete; 32x32 is the verified envelope), so the testbench executor
zero-pads every pass to at least 32x32 — exact for the GEMV, but it means
small heads (`d_k < 32`) and short sequences waste cycles on padding. The
planned hardware phases remove the round-trip and softmax costs:

| Phase | Addition | Removes |
|---|---|---|
| **0** (this chapter) | `AttentionScheduler`, host softmax/requant | — |
| **1** | psum-to-weight-buffer feedback path (dense weight format, transposed write order via address generation) | host round-trips for K, P·V operands |
| **2** | softmax unit in the psum output path (row max, exp LUT, reciprocal, requantize) | host softmax; enables chip-resident stage 2 -> 3 transition |

Phase 1/2 implementations must reproduce the Phase-0 integer results
bit-exactly; the software test doubles as their golden model.
