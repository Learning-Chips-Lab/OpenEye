# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

import sys
import os
import cocotb
import tensorflow as tf
from cocotb.clock import Clock
import open_eye.test_utils_main as tum
import open_eye.rtl_test_utils as rtl_test_utils
import open_eye.timing_parameters as tp
import open_eye.generic_test_utils as gtu
import open_eye.DRAM as DRAM
import open_eye.time_stamper as time_stamper
import open_eye.open_eye_parameters as oep
import open_eye.layer_parameters as lp
import open_eye.simple_layer_operations as slo
import open_eye.layer_execution_state as les
import open_eye.data_create as data_create
import open_eye.tflite2model as tflite2model
from cocotb.logging import SimLogFormatter
import logging
import sys
import logging
from cocotb.utils import get_sim_time
from cocotb.triggers import FallingEdge, RisingEdge, Timer, with_timeout, SimTimeoutError
import numpy as np
import tensorflow as tf

# 1. Keras-Modell definieren
base_model = tf.keras.applications.MobileNet(
    input_shape=(128, 128, 3), alpha=0.50, include_top=True, weights='imagenet'
)

# 2. Zu TFLite konvertieren (Gewichte auf INT8 quantisieren)
converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_bytes = converter.convert()
print(dir(tflite_model_bytes))

# 3. TFLite Interpreter laden, um an die echten INT8-Bytes zu kommen
interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
interpreter.allocate_tensors()
tensor_details = interpreter.get_tensor_details()

# 4. Iterativ durch die Keras-Layer gehen
for layer in base_model.layers:
    if "Conv2D" in type(layer).__name__:
        print(f"=== Suche quantisierte Gewichte für Layer: {layer.name} ===")
        
        # Wir suchen im TFLite-Modell nach dem Tensor, der den Keras-Namen enthält
        # und ein Gewichts-Tensor (weight) ist.
        tflite_tensor = None
        for tensor in tensor_details:
            # TFLite benennt die Tensoren oft "Unterschiedlich/Keras_Layer_Name/..."
            if layer.name in tensor['name'] and ("weight" in tensor['name'].lower() or "kernel" in tensor['name'].lower()):
                tflite_tensor = tensor
                break
                
        if tflite_tensor is not None:
            # Die echten, quantisierten INT8-Gewichte auslesen!
            quantized_weights = interpreter.get_tensor(tflite_tensor['index'])
            
            # Für FPGAs extrem wichtig: Scale und Zero-Point extrahieren
            # Da es sich um Conv-Gewichte handelt, ist es oft "per-channel" quantisiert (Vektor aus Scales)
            quant_params = tflite_tensor['quantization_parameters']
            scales = quant_params['scales']
            zero_points = quant_params['zero_points']
            
            print(f"Erfolgreich im TFLite-Modell gefunden!")
            print(f"Tensor-Name im TFLite-Graph: {tflite_tensor['name']}")
            print(f"Datentyp im Speicher:         {quantized_weights.dtype}") # Sollte int8 oder qint8 sein
            print(f"Gewichte-Shape (TFLite):      {quantized_weights.shape} -> [Out, H, W, In]")
            print(f"Quantisierung - Zero-Point:   {zero_points}")
            print(f"Quantisierung - Erster Scale:  {scales[0] if len(scales) > 0 else 'N/A'}")
            print("-" * 50)
            
            # Hier kannst du die `quantized_weights` (reine INT8-Arrays) direkt in 
            # deine Cocotb-Logik einspeisen oder als Text/Hex-Datei abspeichern.
            
        else:
            print(f"WARNUNG: Konnte quantisierte Gewichte für {layer.name} nicht im TFLite-Modell finden.")
            
        # Da du nur den ersten Layer suchst, brechen wir nach der ersten Convolution ab
        break