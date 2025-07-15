# Getting started

These tutorials show how the OpenEye can be simulated as standalone Verilog code.
We assume that a proper Verilog simulator and all Python requirements (according to the `requirements.txt` have been installed).

## Example Use Case 1: MNIST using Tensorflow/Keras

### Download the data and train the model

You can use the following code to download the data and train the model:

....

### Convert the model to TFLite and quantize it to 8 bit integer

...

### Simulate the OpenEye classification using the trained and quantized model



## Example Use Case 2: MobileNet

**Todo**

## Example Use Case 3: ResNet

**Todo**

## Example Use Case 4: Human Activity Recognition

**Todo**

## Example Use Case 5: EEGNet

**Todo**

# FPGA usage

The followin tutorials show how the OpenEye can be used as an IP-Core inside a block design for AMD FPGAs. 
We use the MNIST model from above and deploy it on an FPGA.

The Xilinx UltraScale XCZU-19EG was used for the reference design located in ... 

## Part 1: Creation of the block design

....

## Part 2: ... 