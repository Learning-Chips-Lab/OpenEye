# OpenEye 
| ![OpenEyeLogo](/doc/figures/open_eye_logo.png) | **The Open Source Hardware Accelerator for Efficient Neural Network Inference** |
| - | - |

## About the project

OpenEye is an open source hardware accelerator for DNN inference. OpenEye is inspired by the [EyerissV2](https://arxiv.org/pdf/1807.07928). Particularly, it adopts key features such as configurability and scalability, ensuring efficient performance across different hardware configurations. Additionally, it leverages sparsity at all hierarchical levels, optimizing the use of resources by minimizing unnecessary computations. The project also implements the *Row-Stationary Dataflow*, which enhances data reuse and reduces energy consumption. Moreover, OpenEye offers high flexibility, allowing it to adapt to various applications and workloads with ease.

## Getting started

The folder [`test`](test) contains different testbenches that can be used to simulate
specific parts oder functionalities of Open Eye.

To simulate a testbench using Icarus Verilog, you can use the [Makefile](test/cocotb_parallel/Makefile). This will simulate a single layer. The [Makefile](test/cocotb_parallel/Makefile) can be adapted to test other parameters of a given layer. Using this will result in a VCD, that can be inspected by a wave viewer (e.g. [GTKWave](https://gtkwave.sourceforge.net/)).

To test complete nets, we designed testbenches, that can be run with [pytest](https://docs.pytest.org/en/stable/) and [cocotb](https://www.cocotb.org/). These nets are [AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), [ResNet](https://arxiv.org/pdf/1512.03385) and [MobileNet](https://arxiv.org/pdf/1704.04861). 

## Documentation

You can find the documentation regarding general properties, design decisions as well as the module documentation [here](TODO:LINK!)

## Requirements

The requirements for the pytests can be installed using the [requirements.txt](requirements.txt).

The simulation can be done with either Icarus Verilog or Verilator. In order to get Verilator to work, we had to patch `include/verilatedos.h`:
```diff
#ifndef VL_VALUE_STRING_MAX_WORDS
-    #define VL_VALUE_STRING_MAX_WORDS 64  ///< Max size in words of String conversion operation
+    #define VL_VALUE_STRING_MAX_WORDS 128  ///< Max size in words of String conversion operation
#endif
```

## Licensing

OpenEye is covered by the Solderpad License, Version 2.1 (see [LICENSE](LICENSE) for full text).

