# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import onnx
import numpy as np
import h5py
from pathlib import Path
import requests


# Laden des ONNX-Modells
script_dir = Path(__file__).resolve().parent
onnx_model_path = Path.joinpath(script_dir, 'ssd_mobilenet_v1_12-int8.onnx')

if not onnx_model_path.exists():
    mobilenet_v1_url = 'https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12-int8.onnx'
    
    response = requests.get(mobilenet_v1_url)
    if response.status_code == 200:
        data = response.content
        onnx_model_path.open('wb').write(data)

onnx_model = onnx.load(onnx_model_path)

# Zugriff auf den Graphen des Modells
graph = onnx_model.graph


layer_w_data = {
    1:("1","Conv__5257_quant", 
        "3,3,3,32",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/Conv2D/merged_input:0_quantized",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/batchnorm/sub:0_quantized"),
    2:("2","FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise_quant",
        "3,3,1,32",
        "ConvMulFusion_W_const_fold_opt__5537_quantized",
        "ConvAddFusion_Add_B_const_fold_opt__5630_quantized"),
    3:("3","Conv__5268_quant",
        "1,1,32,64",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D/merged_input:0_quantized",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/batchnorm/sub:0_quantized"),
    4:("4","FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise_quant",
        "3,3,1,64",
        "ConvMulFusion_W_const_fold_opt__5554_quantized",
        "ConvAddFusion_Add_B_const_fold_opt__5575_quantized"),
    5:("5","Conv__5278_quant",
        "1,1,64,128",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D/merged_input:0_quantized",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/batchnorm/sub:0_quantized"),
    6:("6","FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise_quant",
        "3,3,1,128",
        "ConvMulFusion_W_const_fold_opt__5552_quantized",
        "ConvAddFusion_Add_B_const_fold_opt__5624_quantized"),
    7:("7","Conv__5288_quant",
        "1,1,128,128",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D/merged_input:0_quantized",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/batchnorm/sub:0_quantized"),
    8:("8","FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise_quant",
        "3,3,1,128",
        "ConvMulFusion_W_const_fold_opt__5551_quantized",
        "ConvAddFusion_Add_B_const_fold_opt__5577_quantized"),
    9:("9","Conv__5298_quant",
        "1,1,128,256",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D/merged_input:0_quantized",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/batchnorm/sub:0_quantized"),
    10:("10","FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise_quant",
        "3,3,1,256",
        "ConvMulFusion_W_const_fold_opt__5617_quantized",
        "ConvAddFusion_Add_B_const_fold_opt__5628_quantized"),
    11:("11","Conv__5308_quant",
        "1,1,256,256",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D/merged_input:0_quantized",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/batchnorm/sub:0_quantized"),
    12:("12","FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise_quant",
        "3,3,1,256",
        "ConvMulFusion_W_const_fold_opt__5615_quantized",
        "ConvAddFusion_Add_B_const_fold_opt__5599_quantized"),
    13:("13","Conv__5318_quant",
        "1,1,256,512",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D/merged_input:0_quantized",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/batchnorm/sub:0_quantized"),
    14:("14","FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise_quant",
        "3,3,1,512",
        "ConvMulFusion_W_const_fold_opt__5613_quantized",
        "ConvAddFusion_Add_B_const_fold_opt__5632_quantized"),
    15:("15","Conv__5328_quant",
        "1,1,512,512",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D/merged_input:0_quantized",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/batchnorm/sub:0_quantized"),
    16:("16","FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise_quant",
        "3,3,1,512",
        "ConvMulFusion_W_const_fold_opt__5612_quantized",
        "ConvAddFusion_Add_B_const_fold_opt__5634_quantized"),
    17:("17","Conv__5338_quant",
        "1,1,512,512",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D/merged_input:0_quantized",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/batchnorm/sub:0_quantized"),
    18:("18","FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise_quant",
        "3,3,1,512",
        "ConvMulFusion_W_const_fold_opt__5610_quantized",
        "ConvAddFusion_Add_B_const_fold_opt__5623_quantized"),
    19:("19","Conv__5348_quant",
        "1,1,512,512",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D/merged_input:0_quantized",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/batchnorm/sub:0_quantized"),
    20:("20","FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise_quant",
        "3,3,1,512",
        "ConvMulFusion_W_const_fold_opt__5626_quantized",
        "ConvAddFusion_Add_B_const_fold_opt__5627_quantized"),
    21:("21","Conv__5358_quant",
        "1,1,512,512",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D/merged_input:0_quantized",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/batchnorm/sub:0_quantized"),
    22:("22","FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise_quant",
        "3,3,1,512",
        "ConvMulFusion_W_const_fold_opt__5625_quantized",
        "ConvAddFusion_Add_B_const_fold_opt__5594_quantized"),
    23:("23","Conv__5368_quant",
        "1,1,512,512",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D/merged_input:0_quantized",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/batchnorm/sub:0_quantized"),
    24:("24","FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise_quant",
        "3,3,1,512",
        "ConvMulFusion_W_const_fold_opt__5564_quantized",
        "ConvAddFusion_Add_B_const_fold_opt__5631_quantized"),
    25:("25","Conv__5386_quant",
        "1,1,512,1024",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D/merged_input:0_quantized",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/batchnorm/sub:0_quantized"),
    26:("26","FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise_quant",
        "3,3,1,1024",
        "ConvMulFusion_W_const_fold_opt__5582_quantized",
        "ConvAddFusion_Add_B_const_fold_opt__5636_quantized"),
    27:("27","Conv__5396_quant",
        "1,1,1024,1024",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D/merged_input:0_quantized",
        "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/batchnorm/sub:0_quantized"),
}

def extract_attribute(node, attr_name):
    for attr in node.attribute:
        if attr.name == attr_name:
            if attr.ints:
                return attr.ints
            elif attr.floats:
                return attr.floats
            else:
                return attr.i
    return None


def extract_and_write_info(ID, graph, node_name, dimension, w_ident, b_ident, hdf5_file):
    dimension_list = list(map(int, dimension.split(',')))

    for node in graph.node:
        if node_name in node.name:

            dilations = extract_attribute(node, "dilations")
            group = extract_attribute(node, "group")
            kernel_shape = extract_attribute(node, "kernel_shape")
            pads = extract_attribute(node, "pads")
            strides = extract_attribute(node, "strides")
            filters = dimension_list[-1]

            #print(pads)


            weights_node = next((input for input in node.input if w_ident in input), None)
            bias_node = next((input for input in node.input if b_ident in input), None)

            if weights_node:
                for initializer in graph.initializer:
                    if weights_node in initializer.name:
                        weights_raw_data = np.array(initializer.raw_data)
                        weights = np.frombuffer(weights_raw_data, dtype=np.int8)
                        weights = weights.reshape(dimension_list)
                        hdf5_file.create_dataset(ID + "/weights", data=weights)
                        break

            if bias_node:
                for initializer in graph.initializer:
                    if bias_node in initializer.name:
                        bias_raw_data = np.array(initializer.raw_data)
                        bias = np.frombuffer(bias_raw_data, dtype=np.int32)
                        hdf5_file.create_dataset(ID + "/bias", data=bias)
                        break

            hdf5_file.create_dataset(ID + "/filters", data=filters)            
            hdf5_file.create_dataset(ID + "/dilations", data=dilations)
            hdf5_file.create_dataset(ID + "/group", data=group)
            hdf5_file.create_dataset(ID + "/kernel_shape", data=kernel_shape)
            hdf5_file.create_dataset(ID + "/pads", data=pads)
            hdf5_file.create_dataset(ID + "/strides", data=strides)

            return

    print("Node not found:", node_name)


with h5py.File("model_info.h5", "w") as hdf5_file:
    for key, value in layer_w_data.items():
        ID, node_name, dimension, w_ident, b_ident = value
        extract_and_write_info(ID, graph, node_name, dimension, w_ident, b_ident, hdf5_file)
