# Sparse weight encoding in OpenEye

How a weight matrix becomes a scratchpad image, what each bookkeeping signal in
`hdl/data_pipeline_wght.v` means, and which parts are verified.

## The problem being solved

A PE holds weights for `filters_w` filters (M0) across several rows, where a row
is one spatial/channel term. Written out, the weights form a
`[rows][filters_w]` matrix walked row-major.

Zero weights contribute nothing to a MAC, so they are not transmitted. That
saves bandwidth but destroys the positional information: after compression, the
n-th transmitted weight is no longer the n-th filter. The encoding restores it
by attaching a small **overhead tag** to every transmitted value.

## The contract

Each transmitted sub-word is `{overhead[3:0], payload[7:0]}` (12 bits):

- `payload` - the INT8 weight.
- `overhead` - how many positions were skipped **immediately before** this
  value.

Reconstruction, per row, starting at position 0:

```
position += overhead        # step over the zeros that were dropped
filter    = position        # this value belongs to that filter
position += 1               # this value occupies one position
```

Two consequences worth stating explicitly:

- The tag is a **relative** skip count, not an absolute filter index.
- Positions restart at each row, so rows must be segmented before decoding.
  The segmentation comes from the first SPAD (below), not from the data stream.

`PARALLEL_MACS` sub-words are packed per transmitted word, low lane first. One
second-SPAD word holds all of them: `SECOND_SPAD_DATA = 12 * PARALLEL_MACS`
(24 bits for PARALLEL_MACS = 2).

## Row padding

Rows are packed independently, so a row whose non-zero count is not a multiple
of `PARALLEL_MACS` is padded with all-zero sub-words to start the next row on a
fresh word. A zero payload is unambiguous padding: in sparse mode a zero weight
is never transmitted, so a stored zero cannot be real data.

Padding occupies sub-word slots. Decoding the stream as one flat sequence
therefore mistakes a pad for a weight and shifts every later filter - which is
why row segmentation is not optional.

## The two scratchpads

**Second SPAD (data).** The weights with their corrected tags. Word layout:

```
[SECOND_SPAD_DATA-1 : 12]  higher lanes, passed through from data_i unchanged
[11 : 8]                   overhead_output - corrected tag for lane 0 only
[7 : 0]                    lane 0 payload
```

Note the asymmetry: `premade_spad_2_output` splices the corrected tag into bits
`[11:8]`, so **only lane 0's tag is rewritten**; higher lanes carry the tag the
encoder produced. Any consumer must treat the lanes identically, because both
forms are meant to express the same relative-skip semantics.

**First SPAD (address).** Entry `r` is the cumulative number of second-SPAD
words consumed through the end of weight row `r`. This is how the MAC engine
finds row boundaries. It is written once per input word, re-writing the same
address as a row is consumed, so the meaningful value at an address is the last
one written there.

## Signal glossary

| signal | meaning |
|---|---|
| `overhead_reg` | running position within the current filter row. Advances by `PARALLEL_MACS + overhead_w` per input word: the values consumed plus the zeros they skipped. |
| `overhead_w` | total skips contributed by the sub-words of the current input word (sum of their tags). |
| `overhead_next_word` | look-ahead at sub-word 0's tag of the incoming word, used to detect a row boundary before `overhead_reg` is updated. |
| `overhead_delay_reg` | snapshot of `overhead_reg` taken *before* the current cycle's update. `overhead_output` is a combinational function of `data_storage_2`, which holds the *previous* cycle's data, so it must be compared against the matching earlier position. This is pipeline alignment, not a second counter. |
| `overhead_new_calc_reg` | snapshot of `overhead_reg` for the delayed write path, which decides whether the first SPAD address advances. |
| `over_ending` | set when a row-boundary wrap overshoots by exactly one, and subtracted in `overhead_output` to compensate. |
| `filters_w` | the row length, i.e. filter count M0. Driven from `filters_reg_M0`. Despite the name it is *not* a filter size. |

The wrap in `overhead_output` rebases a tag that would run past the end of a row
so the MAC engine sees a position relative to the current row's start:

```verilog
(tag != 0) && (overhead_delay_reg + tag >= filters_w)
  ? tag + overhead_delay_reg - filters_w - over_ending
  : tag
```

## What is verified

`test/cocotb_data_pipeline/` drives the module directly with known zero patterns
and checks both contracts: that every weight is reconstructed onto its original
filter, and that the first SPAD's cumulative counts mark the right row
boundaries. It covers the dense control, single zeros at the start / interior /
end of a row, zeros in later rows, adjacent zeros, and two zeros within one row.

All of these pass, including the pattern that fails at PE_cluster level
(`WGHT_ZERO_POS="||13,16"`). **The weight pipeline itself is therefore not the
source of the remaining sparse mismatches** - it emits a correct SPAD image, and
the defect lies in how PE.v's compute engine consumes it.
