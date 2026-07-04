# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""OpenEye deployable model artifact (.oeye) builder and reader.

An artifact is a single binary file that contains everything a target
(bare-metal Zynq, Zynq Linux, RISC-V SoC) needs to run one model on the
OpenEye accelerator:

  - one DMA blob per transfer (layer repetition), byte-exact as sent
    over the 64-bit AXI stream
  - per-transfer metadata: length, expected readback bytes, optional
    input-patch window (offset of the iact section inside the blob)
  - model I/O metadata: input/output length and quantization parameters
  - optional embedded reference output for on-target self-test

The C runtime in fpga/sw/runtime/ parses this format. Keep the layout
in sync with openeye_runtime.h (FORMAT_VERSION guards mismatches).

Binary layout (all little-endian, sections 8-byte aligned):

  Header (64 bytes):
      0  u32  magic 'OEYE' (0x4559454F)
      4  u16  version (=1)
      6  u16  header size (=64)
      8  u32  hw_hash        FNV-1a of canonical hardware parameter string
     12  u32  n_transfers
     16  u32  transfer table offset
     20  u32  blob section offset
     24  u32  blob section length
     28  u32  input_len      raw input bytes the runtime may patch (0 = baked)
     32  u32  output_len     bytes returned by the final readback
     36  u32  ref offset     embedded reference output (0 = none)
     40  u32  ref length
     44  f32  input scale
     48  i32  input zero point
     52  f32  output scale
     56  i32  output zero point
     60  u32  reserved

  Transfer record (32 bytes each):
      0  u32  blob offset (relative to blob section)
      4  u32  blob length in bytes (multiple of 8)
      8  u32  out_bytes    expected readback bytes (0 = write only)
     12  u32  flags        bit0: blob contains a patchable input window
     16  u32  patch offset within blob (start of iact section)
     20  u32  patch length in bytes
     24  u16  layer id
     26  u16  repetition
     28  u32  reserved

Typical usage (from the existing demo/ stream directory):

    >>> from open_eye.artifact import ArtifactBuilder
    >>> b = ArtifactBuilder.from_demo_dir('test/cocotb_fpga/demo')
    >>> b.write('model.oeye')

or directly from mapper output (records the iact patch window):

    >>> b = ArtifactBuilder()
    >>> b.add_layer_from_stream(mapper.get_stream(), layer_id=0)
    >>> b.write('model.oeye')
"""

import re
import struct
import logging
from pathlib import Path

import open_eye.stream_dicts as strdic

logger = logging.getLogger("cocotb")

MAGIC = 0x4559454F  # 'OEYE'
FORMAT_VERSION = 1
HEADER_SIZE = 64
TRANSFER_RECORD_SIZE = 32
WORD_BYTES = 8

FLAG_HAS_INPUT_PATCH = 0x1

# Section order used by generic_test_utils.create_stream_file and the
# serial DMA transmission.
_SECTION_ORDER = ["status", "iact", "wght", "psum", "quantize", "offset"]


def fnv1a32(data):
    """FNV-1a 32-bit hash. Mirrors the C implementation in the runtime."""
    h = 0x811C9DC5
    for byte in data:
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def hw_param_hash(params):
    """Hash the hardware parameters that define the stream format.

    A blob generated for one parameterization is invalid on another.
    The canonical string keeps the hash stable across attribute order.
    """
    fields = [
        "Clusters_X", "Clusters_Y", "Iact_Routers", "Wght_Routers",
        "Psum_Routers", "NUM_GLB_IACT", "NUM_GLB_WGHT", "NUM_GLB_PSUM",
        "PEs_X", "PEs_Y", "DATA_IACT_BITWIDTH", "DATA_WGHT_BITWIDTH",
        "DATA_PSUM_BITWIDTH", "SERIAL",
    ]
    parts = []
    for f in fields:
        if hasattr(params, f):
            parts.append(f + "=" + str(getattr(params, f)))
    canonical = ";".join(parts)
    return fnv1a32(canonical.encode("ascii"))


def _words_to_bytes(words):
    """Pack a list of 64-bit integer words little-endian, as the DMA sends them."""
    out = bytearray()
    for w in words:
        out += int(w).to_bytes(WORD_BYTES, "little")
    return bytes(out)


class TransferRecord:
    """One DMA transfer: blob plus metadata."""

    def __init__(self, blob, layer_id=0, repetition=0, out_bytes=0,
                 patch_offset=0, patch_len=0):
        if len(blob) % WORD_BYTES != 0:
            raise ValueError("blob length must be a multiple of 8 bytes")
        self.blob = blob
        self.layer_id = layer_id
        self.repetition = repetition
        self.out_bytes = out_bytes
        self.patch_offset = patch_offset
        self.patch_len = patch_len

    @property
    def flags(self):
        return FLAG_HAS_INPUT_PATCH if self.patch_len else 0


class ArtifactBuilder:
    """Collects transfers and serializes the .oeye container."""

    def __init__(self, hw_hash=0):
        self.transfers = []
        self.hw_hash = hw_hash
        self.input_len = 0
        self.output_len = 0
        self.reference = b""
        self.input_scale = 1.0
        self.input_zero_point = 0
        self.output_scale = 1.0
        self.output_zero_point = 0

    # -- construction -----------------------------------------------------

    def add_layer_from_stream(self, stream, layer_id, repetition=0,
                              out_bytes=0, patchable_input=False):
        """Add a transfer from mapper output (LayerMapper.get_stream(), serial mode).

        Section lengths are known here, so the iact window can be recorded
        for runtime input patching (first layer only, patchable_input=True).
        """
        words = []
        patch_offset = 0
        patch_len = 0
        for name in _SECTION_ORDER:
            section = stream[strdic.stream_parallel_dict[name]]
            if name == "iact":
                patch_offset = len(words) * WORD_BYTES
                patch_len = len(section) * WORD_BYTES
            words.extend(section)
        rec = TransferRecord(
            _words_to_bytes(words), layer_id=layer_id, repetition=repetition,
            out_bytes=out_bytes,
            patch_offset=patch_offset if patchable_input else 0,
            patch_len=patch_len if patchable_input else 0)
        self.transfers.append(rec)
        return rec

    def add_layer_from_words(self, words, layer_id, repetition=0, out_bytes=0):
        """Add a transfer from a flat word list (no patch metadata available)."""
        rec = TransferRecord(_words_to_bytes(words), layer_id=layer_id,
                             repetition=repetition, out_bytes=out_bytes)
        self.transfers.append(rec)
        return rec

    def set_io(self, input_len=0, output_len=0, input_scale=1.0,
               input_zero_point=0, output_scale=1.0, output_zero_point=0):
        self.input_len = input_len
        self.output_len = output_len
        self.input_scale = input_scale
        self.input_zero_point = input_zero_point
        self.output_scale = output_scale
        self.output_zero_point = output_zero_point

    def set_reference(self, ref_bytes):
        """Embed the expected final output for on-target self-test."""
        self.reference = bytes(ref_bytes)

    @classmethod
    def from_demo_dir(cls, demo_path, hw_hash=0):
        """Build from a test/cocotb_fpga/demo style directory.

        Reads layer_<n>_<r>/dma_stream_input.txt (decimal words) in
        (layer, repetition) order. The last transfer gets the readback,
        sized and referenced from its dma_stream_ref.txt (binary words).
        Patch metadata is not recoverable from the text dumps; use
        add_layer_from_stream for runtime input injection.
        """
        demo_path = Path(demo_path)
        dirs = []
        for d in demo_path.glob("layer_[0-9]*_[0-9]*"):
            m = re.fullmatch(r"layer_([0-9]+)_([0-9]+)", d.name)
            if m and (d / "dma_stream_input.txt").exists():
                dirs.append((int(m.group(1)), int(m.group(2)), d))
        if not dirs:
            raise FileNotFoundError(f"no layer_*_* stream directories in {demo_path}")
        dirs.sort()

        builder = cls(hw_hash=hw_hash)
        for layer_id, repetition, d in dirs:
            with open(d / "dma_stream_input.txt") as f:
                words = [int(line) for line in f if line.strip()]
            builder.add_layer_from_words(words, layer_id, repetition)

        # Reference output of the last transfer defines the readback.
        ref_file = dirs[-1][2] / "dma_stream_ref.txt"
        if ref_file.exists():
            with open(ref_file) as f:
                ref_words = [int(line, 2) for line in f if line.strip()]
            ref = _words_to_bytes(ref_words)
            builder.transfers[-1].out_bytes = len(ref)
            builder.output_len = len(ref)
            builder.set_reference(ref)
        else:
            logger.warning("no dma_stream_ref.txt for last layer; "
                           "readback length unknown")
        return builder

    # -- serialization ----------------------------------------------------

    def write(self, path):
        data = self.to_bytes()
        with open(path, "wb") as f:
            f.write(data)
        logger.info(f"wrote {path}: {len(self.transfers)} transfers, "
                    f"{len(data)} bytes")
        return len(data)

    def to_bytes(self):
        if not self.transfers:
            raise ValueError("no transfers added")

        def align8(n):
            return (n + 7) & ~7

        table_off = HEADER_SIZE
        table_len = len(self.transfers) * TRANSFER_RECORD_SIZE
        blob_off = align8(table_off + table_len)

        blobs = bytearray()
        records = []
        for t in self.transfers:
            records.append((len(blobs), t))
            blobs += t.blob

        ref_off = 0
        ref_pos = align8(blob_off + len(blobs))
        if self.reference:
            ref_off = ref_pos

        header = struct.pack(
            "<IHHIIIIIIIIIfifiI",
            MAGIC, FORMAT_VERSION, HEADER_SIZE,
            self.hw_hash, len(self.transfers),
            table_off, blob_off, len(blobs),
            self.input_len, self.output_len,
            ref_off, len(self.reference),
            self.input_scale, self.input_zero_point,
            self.output_scale, self.output_zero_point,
            0)
        assert len(header) == HEADER_SIZE

        table = bytearray()
        for blob_offset, t in records:
            table += struct.pack(
                "<IIIIIIHHI",
                blob_offset, len(t.blob), t.out_bytes, t.flags,
                t.patch_offset, t.patch_len,
                t.layer_id, t.repetition, 0)

        out = bytearray(header)
        out += table
        out += b"\x00" * (blob_off - len(out))
        out += blobs
        if self.reference:
            out += b"\x00" * (ref_pos - len(out))
            out += self.reference
        return bytes(out)


class Artifact:
    """Reader for .oeye files (verification and inspection)."""

    def __init__(self, data):
        if len(data) < HEADER_SIZE:
            raise ValueError("file too small")
        (magic, version, header_size, self.hw_hash, n_transfers,
         table_off, blob_off, blob_len,
         self.input_len, self.output_len, ref_off, ref_len,
         self.input_scale, self.input_zero_point,
         self.output_scale, self.output_zero_point,
         _reserved) = struct.unpack_from("<IHHIIIIIIIIIfifiI", data, 0)
        if magic != MAGIC:
            raise ValueError("bad magic")
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported version {version}")
        if header_size != HEADER_SIZE:
            raise ValueError("bad header size")

        self.transfers = []
        for i in range(n_transfers):
            (b_off, b_len, out_bytes, flags, p_off, p_len,
             layer_id, repetition, _res) = struct.unpack_from(
                "<IIIIIIHHI", data, table_off + i * TRANSFER_RECORD_SIZE)
            blob = data[blob_off + b_off: blob_off + b_off + b_len]
            rec = TransferRecord(blob, layer_id, repetition, out_bytes,
                                 p_off if flags & FLAG_HAS_INPUT_PATCH else 0,
                                 p_len if flags & FLAG_HAS_INPUT_PATCH else 0)
            self.transfers.append(rec)

        self.reference = bytes(data[ref_off:ref_off + ref_len]) if ref_off else b""

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return cls(f.read())

    def summary(self):
        lines = [
            f"format v{FORMAT_VERSION}, hw_hash=0x{self.hw_hash:08x}",
            f"transfers: {len(self.transfers)}",
            f"input_len={self.input_len} output_len={self.output_len}",
            f"input q: scale={self.input_scale} zp={self.input_zero_point}",
            f"output q: scale={self.output_scale} zp={self.output_zero_point}",
            f"reference: {len(self.reference)} bytes",
        ]
        for i, t in enumerate(self.transfers):
            patch = (f" patch@{t.patch_offset}+{t.patch_len}"
                     if t.patch_len else "")
            lines.append(f"  [{i}] layer {t.layer_id}.{t.repetition}: "
                         f"{len(t.blob)} B, readback {t.out_bytes} B{patch}")
        return "\n".join(lines)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Build or inspect OpenEye .oeye model artifacts")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="build from a demo stream directory")
    p_build.add_argument("demo_dir")
    p_build.add_argument("-o", "--output", default="model.oeye")
    p_build.add_argument("--hw-hash", type=lambda v: int(v, 0), default=0)

    p_info = sub.add_parser("info", help="inspect an artifact")
    p_info.add_argument("artifact")

    args = parser.parse_args(argv)
    if args.cmd == "build":
        builder = ArtifactBuilder.from_demo_dir(args.demo_dir,
                                                hw_hash=args.hw_hash)
        size = builder.write(args.output)
        print(f"{args.output}: {size} bytes, {len(builder.transfers)} transfers")
    elif args.cmd == "info":
        print(Artifact.load(args.artifact).summary())


if __name__ == "__main__":
    main()
