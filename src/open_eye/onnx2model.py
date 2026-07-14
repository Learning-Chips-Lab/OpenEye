import os
import torch
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import numpy as np
from onnxruntime.quantization import quant_pre_process, quantize_dynamic, QuantType

class TensorShape:
    """Helper class to allow reading shapes via .shape property, similar to TensorFlow."""
    def __init__(self, shape_list):
        # Swap NCHW to NHWC if the shape belongs to a 4D image/feature tensor
        if len(shape_list) == 4:
            # shape_list layout: [batch_size, channels, height, width]
            # target layout:     [batch_size, height, width, channels]
            self.shape = [shape_list[0], shape_list[2], shape_list[3], shape_list[1]]
        else:
            self.shape = shape_list

    def __repr__(self):
        return str(self.shape)

class HardwareLayer:
    """Represents an extracted layer as a configurable hardware object."""
    def __init__(self, index, layer_type, onnx_op, node_name, has_relu, input_shape, output_shape):
        self.index = index
        self.type = layer_type            # Type name (e.g., conv2d, dense, pooling2d)
        self.name = node_name            # Node name within the ONNX graph
        self.onnx_op = onnx_op
        self.has_relu = has_relu
        
        # Nested shape objects convert NCHW arrays automatically to NHWC format
        self.input = TensorShape(input_shape)
        self.output = TensorShape(output_shape)
        self.kernel = TensorShape([0, 0]) 
        
        # Default structural attributes to prevent subscriptable/NoneType crashes
        self.kernel_size = [0, 0]
        self.strides = [1, 1]
        self.padding_pads = [0, 0, 0, 0]
        self.filters = 0                  
        
        # Channel and feature configurations
        self.out_channels = None
        self.in_channels = None
        self.out_features = None
        self.in_features = None
        
        # TensorFlow convention compatibility: weights[0] = Weights, weights[1] = Bias
        self.weights = [None, None]
        self.bias_fp32 = None
        self.bias_int32 = None
        self.scale = None
        self.zero_point = None

    def __repr__(self):
        return f"<HardwareLayer [{self.index}] {self.type} (Name: {self.name})>"


def export_to_onnx(model, path="simple_mnist_convnet.onnx"):
    """Stable export of PyTorch model into ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model, dummy_input, path,
        export_params=True, opset_version=18, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False
    )
    print(f"Model successfully exported to {path}!")
    return path

def quantize_to_int8(src_path, dest_path="simple_mnist_convnet_int8.onnx"):
    """Executes graph pre-processing and dynamic quantization to signed INT8."""
    prep_path = "simple_mnist_processed.onnx"
    quant_pre_process(input_model_path=src_path, output_model_path=prep_path)
    quantize_dynamic(
        model_input=prep_path, 
        model_output=dest_path, 
        weight_type=QuantType.QInt8
    )
    if os.path.exists(prep_path):
        os.remove(prep_path)
    print(f"Model successfully saved as SIGNED INT8: {dest_path}")
    return dest_path

def build_graph_helpers(onnx_model):
    """Generates maps for look-ahead (ReLU fusion) and quick initializer lookups."""
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    inits = {i.name: numpy_helper.to_array(i) for i in onnx_model.graph.initializer}
    
    next_node_map = {}
    for node in onnx_model.graph.node:
        for inp in node.input:
            next_node_map[inp] = node
            
    return onnx_model, inits, next_node_map

def get_tensor_shapes(onnx_model):
    """Extracts all automatically inferred intermediate tensor shapes."""
    shapes = {}
    graph = onnx_model.graph
    for val in list(graph.value_info) + list(graph.input) + list(graph.output):
        if val.type.tensor_type.HasField("shape"):
            shapes[val.name] = [
                d.dim_value if d.HasField("dim_value") else d.dim_param 
                for d in val.type.tensor_type.shape.dim
            ]
    return shapes

def check_fused_relu(node, next_node_map):
    """Looks ahead in the dataflow to check if the next layer is a ReLU."""
    current_tensor = node.output[0]
    for _ in range(5):
        if current_tensor in next_node_map:
            next_node = next_node_map[current_tensor]
            if next_node.op_type == 'Relu':
                return True
            current_tensor = next_node.output[0]
        else:
            break
    return False

def extract_conv_params(node, w_arr, attrs):
    """Extracts specific structural hyperparameters for a Convolution."""
    return {
        "kernel_shape": [w_arr.shape[2], w_arr.shape[3]],
        "strides": attrs.get("strides", [1, 1]),
        "padding_pads": attrs.get("pads", [0, 0, 0, 0]),
        "out_channels": w_arr.shape[0],
        "in_channels": w_arr.shape[1]
    }

def process_weights_and_biases(layer_obj, node, inits, input_scale, next_node_map):
    """Extracts parameters and locates the split bias tensor from global initializers."""
    w_name = node.input[1]
    if w_name not in inits:
        return layer_obj
    
    raw_weights = inits[w_name]
    layer_obj.zero_point = 0 
    
    # Extract base name (e.g., "Conv1" from "Conv1.weight_quantized")
    base_name = w_name.split('.')[0]
    
    # 1. Target scale and zero-point retrieval
    for s_name in [f"{w_name}_scale", f"{base_name}.weight_scale", f"{base_name}_scale"]:
        if s_name in inits:
            layer_obj.scale = float(inits[s_name])
            break
    for zp_name in [f"{w_name}_zero_point", f"{base_name}.weight_zero_point"]:
        if zp_name in inits:
            layer_obj.zero_point = int(inits[zp_name])

    # 2. Direct bias lookup using PyTorch naming convention patterns
    possible_biases = [f"{base_name}.bias", f"{base_name}_bias", base_name]
    for b_name in possible_biases:
        if b_name in inits:
            layer_obj.bias_fp32 = inits[b_name]
            break

    # 3. Hardware INT32 bias transformation formula
    if layer_obj.bias_fp32 is not None and layer_obj.scale is not None:
        scale_effective = input_scale * layer_obj.scale
        layer_obj.bias_int32 = np.round(
            layer_obj.bias_fp32 / scale_effective
        ).astype(np.int32)
        
    # 4. Transpose weights to match target extraction sequence [x][y][c][f]
    if layer_obj.type == 'conv2d':
        # ONNX layout:  [f, c, y, x] (out_channels, in_channels, height, width)
        # Desired parsing structure: [x][y][c][f]
        layer_obj.weights[0] = np.transpose(raw_weights, (3, 2, 1, 0))
    else:
        layer_obj.weights[0] = raw_weights
        
    layer_obj.weights[1] = layer_obj.bias_int32
        
    return layer_obj

def build_layer_dict(idx, node, shapes, next_node_map):
    """Instantiates a new HardwareLayer object instead of a raw dictionary."""
    op_mapping = {
        'ConvInteger': 'conv2d', 'QLinearConv': 'conv2d',
        'MatMulInteger': 'dense', 'QLinearMatMul': 'dense',
        'MaxPool': 'pooling2d'
    }
    l_type = op_mapping[node.op_type]
    
    return HardwareLayer(
        index=idx,
        layer_type=l_type,
        onnx_op=node.op_type,
        node_name=node.name,
        has_relu=check_fused_relu(node, next_node_map),
        input_shape=shapes.get(node.input[0], "Unknown"),
        output_shape=shapes.get(node.output[0], "Unknown")
    )

def parse_pipeline(onnx_model, inits, next_node_map, shapes):
    """Iterates through nodes to build the clean object-oriented hardware pipeline."""
    pipeline = []
    input_scale = inits.get('input_scale', 1.0)
    
    for node in onnx_model.graph.node:
        if node.op_type not in ['ConvInteger', 'QLinearConv', 'MatMulInteger', 'QLinearMatMul', 'MaxPool']:
            continue
            
        layer = build_layer_dict(len(pipeline), node, shapes, next_node_map)
        attrs = {a.name: (list(a.ints) if a.type == onnx.AttributeProto.INTS else a.i) for a in node.attribute}
        
        if layer.type in ['conv2d', 'dense']:
            layer = process_weights_and_biases(layer, node, inits, input_scale, next_node_map)
            if layer.type == 'conv2d' and layer.weights[0] is not None:
                params = extract_conv_params(node, inits[node.input[1]], attrs)
                layer.kernel = TensorShape(params["kernel_shape"])
                layer.kernel_size = params["kernel_shape"]
                layer.strides = params["strides"]
                layer.padding_pads = params["padding_pads"]
                layer.out_channels = params["out_channels"]
                layer.in_channels = params["in_channels"]
                layer.filters = params["out_channels"]
            elif layer.type == 'dense' and layer.weights[0] is not None:
                layer.out_features = layer.weights[0].shape[0]
                layer.in_features = layer.weights[0].shape[1]
                layer.filters = layer.weights[0].shape[0]
        elif layer.type == 'pooling2d':
            pool_kernel = attrs.get("kernel_shape", [2, 2])
            layer.kernel = TensorShape(pool_kernel)
            layer.kernel_size = pool_kernel
            layer.strides = attrs.get("strides", [2, 2])
            layer.padding_pads = attrs.get("pads", [0, 0, 0, 0])
            
        pipeline.append(layer)
    return pipeline

def print_summary(pipeline):
    """Prints a structured summary of the custom hardware pipeline."""
    print("\n--- Generated Dynamic Hardware Pipeline (including Tensor Shapes) ---")
    for l in pipeline:
        print(f"[{l.index}] Type: {l.type:<15} | In: {str(l.input.shape):<22} | Out: {str(l.output.shape):<22} | Fused ReLU: {str(l.has_relu)}")

def get_model(torch_model):
    """Main Orchestrator: Controls export, quantization, parsing, and verification."""
    fp32_path = export_to_onnx(torch_model)
    int8_path = quantize_to_int8(fp32_path)
    
    loaded_model = onnx.load(int8_path)
    onnx_model, inits, next_node_map = build_graph_helpers(loaded_model)
    shapes = get_tensor_shapes(onnx_model)
    
    pipeline = parse_pipeline(onnx_model, inits, next_node_map, shapes)
    print_summary(pipeline)
    
    for layer_number in range(len(pipeline)):
        pipeline[layer_number].name = pipeline[layer_number].type
    
    # =========================================================================
    # STABLE HARDWARE REFERENCE GENERATOR (Via Graph Modification)
    # =========================================================================
    raw_fp32_model = onnx.load("simple_mnist_convnet.onnx")
    target_fp32_tensor = None
    for node_fp32 in raw_fp32_model.graph.node:
        if node_fp32.op_type == 'MaxPool':
            target_fp32_tensor = node_fp32.output[0]
            break

    if target_fp32_tensor is not None:
        try:
            new_output = onnx.ValueInfoProto()
            new_output.name = target_fp32_tensor
            raw_fp32_model.graph.output.append(new_output)
            
            debug_model_path = "simple_mnist_convnet_debug_nodes.onnx"
            onnx.save(raw_fp32_model, debug_model_path)
            
            session_fp32 = ort.InferenceSession(debug_model_path)
            test_image_fp32 = np.zeros((1, 1, 28, 28), dtype=np.float32)
            test_image_fp32[0, 0, 5, 5] = 1.0
            input_name_fp32 = session_fp32.get_inputs()[0].name
            
            outputs_fp32 = session_fp32.run([target_fp32_tensor], {input_name_fp32: test_image_fp32})
            fp32_intermediate_values = outputs_fp32[0]
            
            if os.path.exists(debug_model_path):
                os.remove(debug_model_path)
            
            conv2_scale = pipeline[2].scale if pipeline[2].scale is not None else 1.0
            input_for_conv2_int8 = np.round(fp32_intermediate_values / conv2_scale).astype(np.int8)
            
        except Exception as e:
            print(f"\n[Hint] Failed executing intermediate FP32 simulation: {e}")
            
    return pipeline