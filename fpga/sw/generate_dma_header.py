import re
from pathlib import Path

def get_stream_dirs(stream_path):
    # build paths to input stream subdirectories
    stream_dirs = sorted(
        stream_path.glob(f'layer_[0-9]*_0'),
        key=lambda p: int(re.search(r'layer_([0-9]*)_0', p.name).group(1))
    )
    return stream_dirs

def write_input_stream_header(stream_path, output_dir='.'):
    stream_dirs = get_stream_dirs(stream_path)
    n_layers = len(stream_dirs)

    # read all streams
    streams = []
    lengths = []
    total_length = 0
    for d in stream_dirs:
        fn = Path(stream_path) / d / 'dma_stream_input.txt'
        stream = open(fn, 'r').readlines()
        streams.append(stream)
        lengths.append(len(stream)*8)
        total_length += len(stream)*8

    # write number of layers and stream lengths per layer
    header = f'int n_layers = {n_layers};\n'
    header += f'int dma_input_stream_lengths[{n_layers}] = {{'
    for n in range(n_layers):
        header += f'{lengths[n]}, '
    header = header[:-2] + '};\n'

    # write every input stream into one array
    header += f'u8 dma_input_stream[{total_length}] = {{\n'
    for stream in streams:
        for n, v in enumerate(stream):
            for i in range(8):
                vv = (int(v) >> i*8) & 0xff
                header += f'0x{vv:02x}, '
            header += '\n'

    header = header[:-3]
    header += '\n};'
    open('dma_stream.h', 'w').write(header)

def write_output_stream_header(stream_path, output_dir='.'):
    stream_dirs = get_stream_dirs(stream_path)

    # build path to reference file
    fn_out = Path(stream_path) / stream_dirs[-1] / 'dma_stream_ref.txt'
    stream = open(fn_out, 'r').readlines()

    # write stream length and data
    header = f'int dma_output_length = {len(stream)*8};\n'
    header += f'u8 dma_output_reference[{len(stream)*8}] = {{\n'

    for n, v in enumerate(stream):
        for i in range(8):
            vv = (int(v, 2) >> i*8) & 0xff
            header += f'0x{vv:02x}, '
        header += '\n'
    header = header[:-3]
    header += '\n};'
    open('dma_stream_out.h', 'w').write(header)

if __name__ == '__main__':
    stream_path = Path(__file__).resolve().parent.parent.parent / 'test/cocotb_fpga/demo'
    write_input_stream_header(stream_path)
    write_output_stream_header(stream_path)