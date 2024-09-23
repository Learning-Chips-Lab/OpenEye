# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

import os
import shutil
import logging

logger = logging.getLogger("cocotb")

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

def open_ref_txts(params, layer_params, layer, layer_number, dram):
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

def check_results(file_1,file_2):
    # Open the two files in read-only mode
    with open(file_1, 'r') as f1, open(file_2, 'r') as f2:
        # Read the contents of the two files into two lists
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    # Compare the two lists line by line and print any differences
    for i, (line1, line2) in enumerate(zip(lines1, lines2)):
        if line1 != line2:
            logger.error(f'Difference found at line {i + 1}:')
            logger.error(f'ReferenceData: {line1.strip()}')
            logger.error(f'Output Stream: {line2.strip()}')

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