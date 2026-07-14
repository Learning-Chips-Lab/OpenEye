# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Host-orchestrated attention scheduler for OpenEye (Phase 0).

Decomposes a single- or multi-head attention operation

    Attention(X) = concat_h( softmax(Q_h K_h^T / sqrt(d_k)) V_h ) W_O

into a sequence of GEMM passes that each run on the accelerator through the
existing dense/GEMM datapath (see gemm_mapper.py and
doc/source/architecture/output_stationary.md), with softmax and INT8
requantization performed on the host between passes.

Phase-0 contract
----------------
- Every on-chip pass is an exact integer GEMV: the accelerator returns the
  raw psum accumulation ``W @ x`` (biases are zero on the dense path).
- Dynamic operands (K, P, V) are re-injected by the host as the weight
  operand of the next pass. Because the DRAM weight layout is
  ``weights[out][in]``, the score pass S = Q K^T needs **no explicit
  transpose**: using K itself as the weight matrix (out = token j,
  in = channel c) computes S[i][j] = sum_c Q[i][c] * K[j][c].
- Between stages the host requantizes intermediate tensors to INT8 with
  power-of-two shifts and applies softmax. The identical integer arithmetic
  is available on the host (``run_on_host``), so hardware execution can be
  checked for bit-exactness at every pass, not just within a tolerance.
- Scales are tracked so the final integer output can be dequantized and
  compared against a float reference (float_reference()).

What Phase 0 intentionally does NOT do: on-chip softmax and on-chip
feedback of activations into the weight buffers. Those are the Phase 1/2
hardware additions described in doc/source/architecture/attention.md; this
module pins down the exact fixed-point behaviour they must reproduce.

Typical usage (testbench or host runtime):

    sched = AttentionScheduler(x, w_q, w_k, w_v, w_o, num_heads=2)
    for stage in sched.stages():
        for p in stage.passes:
            p.result = execute_gemm_pass(p)   # on-chip or p.expected_int()
        stage.finalize()
    out_int, out_scale = sched.final_output()
"""

import math

import numpy as np

INT8_MIN = -128
INT8_MAX = 127


def requantize_pow2(values, target_max=INT8_MAX, nonzero=False):
    """Requantize an integer tensor to INT8 with a power-of-two shift.

    Chooses the smallest shift so that ``round(v / 2**shift)`` fits into
    [-128, 127] and applies it with round-half-away-from-zero semantics
    (matching a hardware arithmetic shift with rounding). Deterministic on
    integers, so host and (future) hardware implementations can agree
    bit-exactly.

    With ``nonzero=True`` any zero result is replaced by +1. This is
    required for tensors that are re-injected as the *weight* operand of a
    later pass: the serial dense weight stream is zero-compressed and the
    datapath does not tolerate zero weight entries (the DRAM writer applies
    the same +-1 replacement to dense weights). The replacement perturbs
    the value by one LSB of the requantized scale and is part of the
    deterministic fixed-point contract (the host golden model applies it
    identically).

    Returns:
        (int8 ndarray, shift)
    """
    values = np.asarray(values, dtype=np.int64)
    max_abs = int(np.max(np.abs(values))) if values.size else 0
    shift = 0
    while max_abs > target_max << shift:
        shift += 1
    if shift == 0:
        quant = values
    else:
        half = 1 << (shift - 1)
        quant = np.where(values >= 0,
                         (values + half) >> shift,
                         -((-values + half) >> shift))
    quant = np.clip(quant, INT8_MIN, INT8_MAX).astype(np.int64)
    if nonzero:
        quant = np.where(quant == 0, 1, quant)
    return quant, shift


def make_weights_nonzero(w):
    """Replace zero entries of a weight matrix by +1 (hardware constraint).

    The dense weight stream cannot carry zero entries (see
    requantize_pow2). Mirrors DRAM.write_initial_data_to_dram, which does
    the same for randomly generated dense weights.
    """
    w = np.asarray(w, dtype=np.int64)
    return np.where(w == 0, 1, w)


class GemmPass(object):
    """One on-chip GEMV pass: result = weight @ iact (raw integer psums).

    Attributes:
        name: Human-readable pass identifier (e.g. "score/h0/t2").
        weight: int ndarray [N, K] in DRAM layout weights[out][in].
        iact: int ndarray [K], INT8 range.
        result: to be filled by the executor with the captured N outputs;
            expected_int() gives the bit-exact value the hardware must
            produce.
    """

    def __init__(self, name, weight, iact):
        self.name = name
        self.weight = np.asarray(weight, dtype=np.int64)
        self.iact = np.asarray(iact, dtype=np.int64)
        if self.weight.ndim != 2 or self.iact.ndim != 1 \
                or self.weight.shape[1] != self.iact.shape[0]:
            raise ValueError(f"shape mismatch in pass {name}: "
                             f"W{self.weight.shape} x{self.iact.shape}")
        if np.max(np.abs(self.weight)) > INT8_MAX + 1 or \
                np.max(np.abs(self.iact)) > INT8_MAX + 1:
            raise ValueError(f"operands of pass {name} exceed INT8 range")
        self.result = None

    @property
    def k(self):
        return self.weight.shape[1]

    @property
    def n(self):
        return self.weight.shape[0]

    def expected_int(self):
        """Bit-exact integer reference of this pass (W @ x, no bias)."""
        return self.weight @ self.iact


class Stage(object):
    """A group of independent passes plus a host-side finalize step."""

    def __init__(self, name, passes, finalize_fn):
        self.name = name
        self.passes = passes
        self._finalize_fn = finalize_fn
        self.finalized = False

    def finalize(self):
        for p in self.passes:
            if p.result is None:
                raise RuntimeError(f"pass {p.name} has no result yet")
        self._finalize_fn([np.asarray(p.result, dtype=np.int64)
                           for p in self.passes])
        self.finalized = True


class AttentionScheduler(object):
    """Lowers one attention operation into OpenEye GEMM passes (Phase 0).

    Args:
        x: int ndarray [seq_len, d_model], INT8, input activations
            (scale ``s_x``: float value = int * s_x).
        w_q, w_k, w_v: int ndarrays [d_model, d_model], INT8 projection
            weights (scale ``s_w``), applied as X @ W.
        w_o: int ndarray [d_model, d_model], INT8 output projection.
        num_heads: number of attention heads; must divide d_model.
        s_x, s_w: dequantization scales of x and the weight matrices,
            only used for the float reference / final dequantization.
        p_scale: integer range used for the quantized softmax output
            (P_int = round(P * p_scale), scale 1/p_scale).

    Stage sequence produced by stages():
        1. projection : 3 * seq_len passes  -> Q, K, V   (then requant INT8)
        2. score      : heads * seq_len     -> S = Q K^T (then softmax -> P)
        3. context    : heads * seq_len     -> O = P V   (then requant INT8)
        4. output     : seq_len             -> out = O W_O
    """

    def __init__(self, x, w_q, w_k, w_v, w_o, num_heads=1,
                 s_x=1.0, s_w=1.0, p_scale=INT8_MAX):
        self.x = np.asarray(x, dtype=np.int64)
        # Weight-operand tensors must be zero-free (dense weight stream
        # constraint); zeros are replaced by +1 like in the DRAM writer.
        self.w_q = make_weights_nonzero(w_q)
        self.w_k = make_weights_nonzero(w_k)
        self.w_v = make_weights_nonzero(w_v)
        self.w_o = make_weights_nonzero(w_o)
        self.seq_len, self.d_model = self.x.shape
        if self.d_model % num_heads != 0:
            raise ValueError("num_heads must divide d_model")
        self.num_heads = num_heads
        self.d_k = self.d_model // num_heads
        self.s_x = float(s_x)
        self.s_w = float(s_w)
        self.p_scale = int(p_scale)

        # Integer intermediate tensors (INT8 after requantization) + shifts
        self.q8 = self.k8 = self.v8 = None
        self.q_shift = self.k_shift = self.v_shift = 0
        self.s_int = None          # raw scores per head [H, S, S]
        self.p8 = None             # quantized softmax [H, S, S]
        self.o8 = None             # context, INT8 [S, d_model]
        self.o_shift = 0
        self.out_int = None        # final raw output [S, d_model]

    # ------------------------------------------------------------------
    # Stage construction
    # ------------------------------------------------------------------
    def stages(self):
        """Generator over the four stages, in execution order.

        Each yielded Stage must be fully executed (all pass results set)
        and finalized before the next stage is requested, because later
        operands are derived from earlier results.
        """
        yield self._projection_stage()
        yield self._score_stage()
        yield self._context_stage()
        yield self._output_stage()

    def _projection_stage(self):
        passes = []
        # Q[i][o] = sum_c X[i][c] * W[c][o]  ->  DRAM weight[o][c] = W.T
        for kind, w in (("q", self.w_q), ("k", self.w_k), ("v", self.w_v)):
            wt = w.T
            for t in range(self.seq_len):
                passes.append(GemmPass(f"proj_{kind}/t{t}", wt, self.x[t]))

        def finalize(results):
            per = self.seq_len
            q = np.stack(results[0:per])
            k = np.stack(results[per:2 * per])
            v = np.stack(results[2 * per:3 * per])
            # Q stays an iact operand (zeros allowed); K and V become
            # weight operands of the score/context passes (zero-free).
            self.q8, self.q_shift = requantize_pow2(q)
            self.k8, self.k_shift = requantize_pow2(k, nonzero=True)
            self.v8, self.v_shift = requantize_pow2(v, nonzero=True)

        return Stage("projection", passes, finalize)

    def _head(self, tensor, h):
        return tensor[:, h * self.d_k:(h + 1) * self.d_k]

    def _score_stage(self):
        passes = []
        # S_h[i][j] = sum_c Q_h[i][c] * K_h[j][c] -> weight = K_h (no transpose)
        for h in range(self.num_heads):
            k_h = self._head(self.k8, h)
            q_h = self._head(self.q8, h)
            for t in range(self.seq_len):
                passes.append(GemmPass(f"score/h{h}/t{t}", k_h, q_h[t]))

        def finalize(results):
            s = np.stack(results).reshape(
                self.num_heads, self.seq_len, self.seq_len)
            self.s_int = s
            # Host softmax: dequantize scores, scale by 1/sqrt(d_k),
            # then quantize the probabilities to [0, p_scale].
            s_scale = (self.s_x * self.s_w) ** 2 \
                * (1 << self.q_shift) * (1 << self.k_shift)
            logits = s.astype(np.float64) * s_scale / math.sqrt(self.d_k)
            logits = logits - logits.max(axis=-1, keepdims=True)
            e = np.exp(logits)
            p = e / e.sum(axis=-1, keepdims=True)
            # P becomes the weight operand of the context passes: quantize
            # to [1, p_scale] (zero probabilities become the minimum weight,
            # 1/p_scale, to satisfy the zero-free weight constraint).
            self.p8 = np.clip(np.rint(p * self.p_scale),
                              1, self.p_scale).astype(np.int64)

        return Stage("score", passes, finalize)

    def _context_stage(self):
        passes = []
        # O_h[i][d] = sum_j P_h[i][j] * V_h[j][d] -> weight = V_h.T
        for h in range(self.num_heads):
            v_h_t = self._head(self.v8, h).T
            for t in range(self.seq_len):
                passes.append(GemmPass(f"context/h{h}/t{t}",
                                       v_h_t, self.p8[h][t]))

        def finalize(results):
            o = np.stack(results).reshape(
                self.num_heads, self.seq_len, self.d_k)
            # concat heads back to [S, d_model]
            o = np.concatenate([o[h] for h in range(self.num_heads)], axis=1)
            self.o8, self.o_shift = requantize_pow2(o)

        return Stage("context", passes, finalize)

    def _output_stage(self):
        passes = []
        w_o_t = self.w_o.T
        for t in range(self.seq_len):
            passes.append(GemmPass(f"output/t{t}", w_o_t, self.o8[t]))

        def finalize(results):
            self.out_int = np.stack(results)

        return Stage("output", passes, finalize)

    # ------------------------------------------------------------------
    # Results and references
    # ------------------------------------------------------------------
    def final_output(self):
        """Return (int output [S, d_model], dequantization scale)."""
        if self.out_int is None:
            raise RuntimeError("output stage not finalized yet")
        v_scale = self.s_x * self.s_w * (1 << self.v_shift)
        out_scale = (v_scale / self.p_scale) \
            * (1 << self.o_shift) * self.s_w
        return self.out_int, out_scale

    def run_on_host(self):
        """Execute the full schedule on the host (bit-exact integer model).

        Drives the same stage machinery with expected_int() as executor.
        Used by the software test and as the golden model the hardware run
        must match pass for pass.
        """
        for stage in self.stages():
            for p in stage.passes:
                p.result = p.expected_int()
            stage.finalize()
        return self.final_output()

    def float_reference(self):
        """Float attention on the dequantized inputs (numpy)."""
        x = self.x.astype(np.float64) * self.s_x
        w_q = self.w_q.astype(np.float64) * self.s_w
        w_k = self.w_k.astype(np.float64) * self.s_w
        w_v = self.w_v.astype(np.float64) * self.s_w
        w_o = self.w_o.astype(np.float64) * self.s_w
        q, k, v = x @ w_q, x @ w_k, x @ w_v
        heads = []
        for h in range(self.num_heads):
            sl = slice(h * self.d_k, (h + 1) * self.d_k)
            logits = (q[:, sl] @ k[:, sl].T) / math.sqrt(self.d_k)
            logits = logits - logits.max(axis=-1, keepdims=True)
            e = np.exp(logits)
            p = e / e.sum(axis=-1, keepdims=True)
            heads.append(p @ v[:, sl])
        return np.concatenate(heads, axis=1) @ w_o

    def total_passes(self):
        """Number of on-chip GEMV passes in the whole schedule."""
        return (3 + self.num_heads * 2 + 1) * self.seq_len
