import tensorflow as tf
import numpy as np


def representative_data_gen():
    for _ in range(100):
        yield [
            np.random.uniform(
                0,
                255,
                size=(1, 128, 128, 3)
            ).astype(np.float32)
        ]


model = tf.keras.applications.MobileNet(
    input_shape=(128, 128, 3),
    alpha=0.50,
    include_top=True,
    weights="imagenet"
)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_data_gen

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("mobilenet_v1_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("Generated mobilenet_v1_int8.tflite")