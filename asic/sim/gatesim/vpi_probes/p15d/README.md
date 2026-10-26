# P15d — functional verdict of the specify-strip transform (real PDK cells)

Verifies `asic/scripts/pdk_specify_strip.py` end-to-end on REAL IHP SG13G2
cell modules (`sg13g2_dfrbpq_1`, `sg13g2_and2_1`, `sg13g2_IOPadIn` +
`ihp_dff_r` from `sg13g2_primitives_vlt.v`), with the fixed file order
(primitives first, cells+top file after → all cells non-public → 2
scopes for both variants) and the real-build corners
(`--flatten --vpi --timing`, 10 marked wrapper ports, cocotb,
Python-driven clock).

## Verdict (recorded 2026-10-25, container iic-osic-tools_asic)

| variant | specify | FF path (`ffq0_o`) | result |
|---|---|---|---|
| raw (specify blocks present) | 5× SPECIFYIGN | **FAIL** (ffq0_o=0: `delayed_CLK` driven only by the ignored specify block → undriven → folds to 0 in 2-state → FF never clocks) | `TESTS=1 PASS=0 FAIL=1` |
| tf (stripped + `assign delayed_X = X` aliases) | 0 | **PASS** (ffq0_o=1) | `P15D_OK`, `TESTS=1 PASS=1 FAIL=0` |

Both variants: 2 scope rows (TOP + p15dtop), 0 per-instance scopes,
combinational check (and2 chain, bits[9:1]=0x122) PASS in both —
the strip only changes the timing-block-driven signals, as intended.

## Notes

- `data_dma_o[0]` is the END of the NF-deep dfrbpq chain, so the FF
  check reads the first cell's Q through the dedicated `ffq0_o` tap
  (1-cycle capture; an earlier version of this tb read the chain end
  after one edge and failed for that reason — fixed in `p15d_tb.py`).
- `gen_p15d.py [NF=10] [NG=10]` extracts the PDK modules, applies the
  same rename as `run_verilator.sh` (unnamed `ihp_*` instances) and the
  strip transform, and writes `p15d_raw.v` / `p15d_tf.v`.

Run (container): `bash run_p15d.sh` (≈3 min for both variants).
