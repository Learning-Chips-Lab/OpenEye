import tensorflow as tf
import numpy as np
import json
import os


MODEL = "mobilenet_v1_int8.tflite"

EXPORT_DIR = "openeye_export"

os.makedirs(EXPORT_DIR, exist_ok=True)

interpreter = tf.lite.Interpreter(
    model_path=MODEL
)

interpreter.allocate_tensors()

tensor_details = interpreter.get_tensor_details()

##############################################################################
# Export global input tensor
##############################################################################

input_tensor = tensor_details[0]

input_quant = {
    "scale": input_tensor["quantization_parameters"]["scales"].tolist(),
    "zero_point":
        input_tensor["quantization_parameters"]["zero_points"].tolist()
}

with open(
    os.path.join(EXPORT_DIR, "model_input_quant.json"),
    "w"
) as f:
    json.dump(input_quant, f, indent=2)

##############################################################################
# Find all weight tensors
##############################################################################

layer_idx = 0

for tensor in tensor_details:

    shape = tensor["shape"]

    if tensor["dtype"] != np.int8:
        continue

    if len(shape) != 4:
        continue

    weights = interpreter.get_tensor(
        tensor["index"]
    )

    qp = tensor["quantization_parameters"]

    scales = qp["scales"]

    zero_points = qp["zero_points"]

    quant_dim = qp["quantized_dimension"]

    ##########################################################################
    # Save weights
    ##########################################################################

    weight_file = os.path.join(
        EXPORT_DIR,
        f"layer_{layer_idx}_weights.npy"
    )

    np.save(weight_file, weights)

    ##########################################################################
    # Save quantization metadata
    ##########################################################################

    metadata = {
        "tensor_name": tensor["name"],

        "shape": shape.tolist(),

        "weight_scales":
            scales.tolist(),

        "weight_zero_points":
            zero_points.tolist(),

        "quantized_dimension":
            int(quant_dim)
    }

    ##########################################################################
    # Search nearest bias tensor
    ##########################################################################

    bias = None

    current_pos = tensor_details.index(tensor)

    for candidate in tensor_details[current_pos+1:]:

        if candidate["dtype"] != np.int32:
            continue

        if len(candidate["shape"]) != 1:
            continue

        bias = candidate
        break

    if bias is not None:

        bias_data = interpreter.get_tensor(
            bias["index"]
        )

        bias_file = os.path.join(
            EXPORT_DIR,
            f"layer_{layer_idx}_bias.npy"
        )

        np.save(
            bias_file,
            bias_data
        )

        metadata["bias_tensor"] = bias["name"]

        metadata["bias_scales"] = \
            bias["quantization_parameters"][
                "scales"
            ].tolist()

    ##########################################################################
    # Save metadata
    ##########################################################################

    meta_file = os.path.join(
        EXPORT_DIR,
        f"layer_{layer_idx}_quant.json"
    )

    with open(meta_file, "w") as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    print(
        f"Exported layer {layer_idx}: "
        f"{tensor['name']}"
    )

    layer_idx += 1

print()
print("Done.")