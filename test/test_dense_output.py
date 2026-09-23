"""Dense DMA capture against explicit wire-order fixtures, without a simulator."""
import asyncio
import logging
from types import SimpleNamespace as NS

import pytest

from open_eye import rtl_test_utils as rtl
from open_eye import test_utils_main as tum


def capture(monkeypatch, words, columns, count, *, repetition=0, tiles=1):
    per_column = (count + columns - 1) // columns
    layer = NS(used_psum_per_PE=per_column, iact_transmissions_pe=1,
               psum_transmissions_pe=tiles, used_Y_cluster=2,
               psum_output_words=per_column * columns)
    params = NS(Clusters_X=columns, Clusters_Y=2, DATA_PSUM_BITWIDTH=20,
                DMA_BITWIDTH=64, PSUM_Trans_Bitwidth=40)
    class Word(int):
        def __getitem__(self, bits):
            return (self >> bits.stop) & ((1 << (bits.start - bits.stop + 1)) - 1)
    class Signal:
        @property
        def value(self):
            value = words[position[0]]
            return Word(value) if isinstance(value, int) else value
    class Enable:
        @property
        def value(self):
            return int(position[0] < len(words))
    position = [0]
    async def timer(*args, **kwargs):
        position[0] += 1
    monkeypatch.setattr(rtl, "Timer", timer)
    monkeypatch.setattr(rtl.cocotb, "start_soon", lambda coro: coro.close())
    monkeypatch.delenv("DUMP_PSUM_BUFFERS", raising=False)
    dut = NS(data_dma_o=Signal(), enable_dma_o=Enable(), ready_dma_i=NS(),
             _log=logging.getLogger("dense_capture_test"))
    dram = NS(fmap=[None, [None] * (count * tiles)])
    asyncio.run(rtl.compare_stream_Dense(
        NS(clk_cycle=20, clk_cycle_unit="ns"), dut, 0, repetition,
        layer, params, NS(), dram, logging.INFO))
    return dram.fmap[1]


@pytest.mark.parametrize("columns,values,wire_order", [
    # The reported failure retained even-numbered outputs, then zeros.
    (1, [17, -4516, 5251, -9954, -7288, -10952, -2542, 4140, -3736, 29],
        [17, -4516, 5251, -9954, -7288, -10952, -2542, 4140, -3736, 29]),
    (2, [0, -1, 524287, -524288, 42, -7], [0, -524288, -1, 42, 524287, -7]),
    (3, [11, 12, 13, 14, 15, 16], [11, 13, 15, 12, 14, 16]),
    # An odd filter count leaves a padding slot in the last column.
    (2, [11, 12, 13, 14, 15], [11, 14, 12, 15, 13, 0]),
])
def test_dense_capture_wire_order(monkeypatch, columns, values, wire_order):
    words = [value & ((1 << 20) - 1) for value in wire_order]
    assert capture(monkeypatch, words, columns, len(values)) == values


@pytest.mark.parametrize("words", [[1, 2, 3], [1, 2, 3, 4, 5]])
def test_dense_capture_rejects_wrong_length(monkeypatch, words):
    with pytest.raises(AssertionError, match="Dense DMA"):
        capture(monkeypatch, words, 1, 4)


def test_dense_capture_rejects_unknown_data(monkeypatch):
    with pytest.raises(ValueError):
        capture(monkeypatch, ["xxxx", 2], 1, 2)


def test_dense_capture_output_tile_offset(monkeypatch):
    assert capture(monkeypatch, [11, 13, 12, 14], 2, 4,
                   repetition=1, tiles=2) == [None] * 4 + [11, 12, 13, 14]


@pytest.mark.parametrize("columns,expected", [
    (1, [11, -12, 13, -14, 15]),
    (2, [11, -14, -12, 15, 13, 0]),
    (3, [11, 13, 15, -12, -14, 0]),
])
def test_dense_serial_reference_words(monkeypatch, tmp_path, columns, expected):
    monkeypatch.chdir(tmp_path)
    for name in ("write_weight_file", "write_iact_file", "write_psum_file"):
        monkeypatch.setattr(tum, name, lambda *args: None)
    params = NS(SERIAL=1, Clusters_X=columns, DMA_BITWIDTH=64, DATA_PSUM_BITWIDTH=20)
    layer = NS(layer_name="Dense", used_psum_per_PE=(5 + columns - 1) // columns,
               needed_total_transmissions=1, iact_transmissions_pe=1,
               psum_transmissions_pe=1)
    tum.make_ref(params, layer, 0, None, dict(enumerate([11, -12, 13, -14, 15])))
    lines = (tmp_path / "demo/layer_0_0/dma_stream_ref.txt").read_text().splitlines()
    assert len(lines) == len(expected)
    assert [int(line, 2) for line in lines] == [v & ((1 << 20) - 1) for v in expected]
    assert all(len(line) == 64 for line in lines)
