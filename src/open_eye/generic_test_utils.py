# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

import os
import shutil
import logging
import open_eye.stream_dicts as strdic

logger = logging.getLogger("cocotb")

def load_env_to_variable(variable_string, default_value):
    val = os.environ.get(variable_string)
    
    if val is None:
        logger.debug(f"{variable_string} not set, using default: {default_value}")

        return default_value
    
    try:
        if isinstance(default_value, int):
            return int(val)
        return val
    except ValueError:
        return default_value
def delete_files_in_directory(directory_path):
    try:
        with os.scandir(directory_path) as entries:
            for entry in entries:
                if entry.is_file():
                    os.unlink(entry.path)
                else:
                    shutil.rmtree(entry.path)
            logger.debug("All files and subdirectories deleted successfully.")
    except OSError:
        logger.debug("Error occurred while deleting files and subdirectories.")

def to_twos_complement(value, bits):
    if value < 0:
        value = (1 << bits) + value
    return value

def to_twos_complement_string(value, bits):
    value = to_twos_complement(value, bits)
    binary_string = format(value, '0' + str(bits) + 'b')
    return binary_string

def twos_complement(binary_str, bits):
    val = int(binary_str, 2)
    # Wenn das MSB (Most Significant Bit) gesetzt ist
    if val & (1 << (bits - 1)):
        val = val - (1 << bits)
    return val

def open_ref_txts(layer_params, layer, layer_number):
    file_dma_ref = [0 for layer_repetition in range(layer_params.needed_total_transmissions)]
    for layer_repetition in range(layer_params.needed_total_transmissions):
        file_dma_ref[layer_repetition] = open_or_create_file('demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/dma_stream_ref.txt')
    
    if "Conv" in str(layer):
        iact_ref = [0 for c in range(layer.input.shape[3])]
        for c in range(layer.input.shape[3]):
            iact_ref[c] = open_or_create_file('demo/layer_' + str(layer_number) + '/iact/iact_ref' + '_' +  str(c) + '.csv')
    elif "Dense" in str(layer):
        iact_ref = open_or_create_file('demo/layer_' + str(layer_number) + '/iact/iact_ref' + '_0.csv')

    if "Depthwise" in str(layer):
        wght_ref = [0  for c in range(layer.input.shape[3])]
        for c in range(layer.input.shape[3]):
            wght_ref[c] = open_or_create_file('demo/layer_' + str(layer_number) + '/weight/wght_ref' + '_' + str(c) + '.csv')

    elif "Conv" in str(layer):
        wght_ref = [[0 for f in range(layer.filters)] for c in range(layer.input.shape[3])]
        for c in range(layer.input.shape[3]):
            for f in range(layer.filters):
                wght_ref[c][f] = open_or_create_file('demo/layer_' + str(layer_number) + '/weight/wght_ref' + '_' + str(c) + '_' + str(f) + '.csv')

    elif "Dense" in str(layer):
        wght_ref = open_or_create_file('demo/layer_' + str(layer_number) + '/weight/wght_ref' + '_0.csv')

    if "Conv" in str(layer):
        psum_ref = [0 for f in range(layer.output.shape[3])]
        for f in range(layer.output.shape[3]):
            psum_ref[f] = open_or_create_file('demo/layer_' + str(layer_number) + '/psum/psum_ref' + '_' +  str(f) + '.csv')

    elif "Dense" in str(layer):
        psum_ref = open_or_create_file('demo/layer_' + str(layer_number) + '/psum/psum_ref' + '_0.csv')

    return file_dma_ref, iact_ref, wght_ref, psum_ref

def open_or_create_file(filepath):
    filename = filepath
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    open_file = open(filename, 'w')
    return open_file

def check_results(params, file_1,file_2):
    # Open the two files in read-only mode
    with open(file_1, 'r') as f1, open(file_2, 'r') as f2:
        # Read the contents of the two files into two lists
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    # Compare the two lists line by line and print any differences
    words = (params.DMA_BITWIDTH//params.DATA_PSUM_BITWIDTH)
    line1_words = [0 for _ in range (words)]
    line2_words = [0 for _ in range (words)]
    error_line1 = ""
    error_line2 = ""
    def decode_words(line):
        """Split a packed line into signed psum values.

        Lines can be shorter than words*DATA_PSUM_BITWIDTH (a truncated or
        empty output line is exactly the failure worth reporting), so decode
        only the slices that are full width instead of letting int('', 2)
        raise and hide the comparison result behind a ValueError.
        """
        stripped = line.strip()
        values = []
        for x in range(words):
            chunk = stripped[x*params.DATA_PSUM_BITWIDTH:(x+1)*params.DATA_PSUM_BITWIDTH]
            if len(chunk) == params.DATA_PSUM_BITWIDTH and set(chunk) <= {"0", "1"}:
                values.append(str(twos_complement(chunk, params.DATA_PSUM_BITWIDTH)))
            else:
                values.append("<{}>".format(chunk if chunk else "missing"))
        return values

    expected_len = words * params.DATA_PSUM_BITWIDTH
    for i, (line1, line2) in enumerate(zip(lines1, lines2)) :
        if line1 != line2:
            ref, out = line1.strip(), line2.strip()
            # An all-zero reference line is padding and not a real difference.
            if set(ref) <= {"0"} and ref:
                continue
            logger.error(f'Difference found at line {i + 1}:')
            if len(ref) != expected_len or len(out) != expected_len:
                logger.error("Line width mismatch: reference %d chars, output %d chars, "
                             "expected %d (%d words x %d bits)",
                             len(ref), len(out), expected_len, words, params.DATA_PSUM_BITWIDTH)
            logger.error("Reference Data: %s   %s", ref, " ".join(decode_words(line1)))
            logger.error("Output Data   : %s   %s", out, " ".join(decode_words(line2)))
            return False
        
    logger.debug('No differences found between files')
    return True

def select_gpu(gpu_id):
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        except RuntimeError as e:
                logger.debug(e)

def reset_nested_list(lst):
    if isinstance(lst, list):
        return [reset_nested_list(sublist) for sublist in lst]
    else:
        return 0

def create_stream_file(stream, layer_number, layer_repetition):
    f_dump = open_or_create_file('demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/dma_stream_input.txt')
    for v in stream[strdic.stream_parallel_dict["status"]]:
        f_dump.write(f'{v}\n')
    for v in stream[strdic.stream_parallel_dict["iact"]]:
        f_dump.write(f'{v}\n')
    for v in stream[strdic.stream_parallel_dict["wght"]]:
        f_dump.write(f'{v}\n')
    for v in stream[strdic.stream_parallel_dict["psum"]]:
        f_dump.write(f'{v}\n')
    for v in stream[strdic.stream_parallel_dict["quantize"]]:
        f_dump.write(f'{v}\n')
        
    f_dump.close()

def transform_n_to_m_chunked(input_list, n, m, chunk_size):
    result = []
    bit_buffer = 0
    buffer_length = 0
    mask_n = (1 << n) - 1
    mask_m = (1 << m) - 1
    chunk_counter = 0
    for value in input_list:
        clean_value = value & mask_n
        bit_buffer |= (clean_value << buffer_length)
        buffer_length += n
        chunk_counter += 1

        while buffer_length >= m:
            block_mbit = bit_buffer & mask_m
            result.append(block_mbit)
            bit_buffer >>= m
            buffer_length -= m

        if chunk_counter == chunk_size:
            if buffer_length > 0:
                result.append(bit_buffer & mask_m)
                bit_buffer = 0
                buffer_length = 0
            chunk_counter = 0

    if buffer_length > 0:
        result.append(bit_buffer & mask_m)

    return result

class HDF5_Model:
    def save_model_to_hdf5(self, model, filename):
        model.save(filename)

    def load_model_from_hdf5(self, filename):
        import tensorflow as tf
        return tf.keras.models.load_model(filename)

    def create_and_save_model(self):
        import tensorflow as tf
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(8, (3, 3), padding="SAME", input_shape=(28, 28, 8), strides = 1))
        self.save_model_to_hdf5(model, '28283_model.h5')
        return model