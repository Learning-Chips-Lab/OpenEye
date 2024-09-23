.. Open Eye documentation master file, created by
   sphinx-quickstart on Mon Jan 22 09:47:36 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Open Eye's documentation!
====================================

OpenEye is an open source hardware accelerator for deep learning. It is
designed to be a flexible platform for research in the field of deep
learning accelerators based on the OpenEye V2 architecture. It is written in
Verilog and can be synthesized to run on an FPGA or ASIC.

In deep learning, there is a trend to from wide and shallow networks to
more narrow and deeper networks. This trend has led to a significant increase in
the number of shapes and sizes of the convolutional kernels and an increase in the
use of bottleneck layers. Furthermore, filter decomposition, pruning and quantization
are becoming more common.

This has made it difficult to design a fixed architecture
that can support a wide range of deep learning models. OpenEye is designed to be
to support exactly these trends.

It has the following features:

* A flexible architecture that can be configured to support a wide range of deep learning models.
* It uses a row stationary dataflow to maximize the utilization of the on-chip memory.
* It supports a wide range of data types and precisions.
* It is designed to be scalable to support a wide range of model sizes.
* It can exploit the sparsity in the model to reduce the energy consumption.

See the `architecture` section for more details on the architecture of the OpenEye.
The `verilog` section contains the documentation for the Verilog code.
See the `test` section for the testbench and the test cases.

Verilog Module Documentation
----------------------------
To generate the documentation for a Verilog module, you have to preprocess the
Verilog files using the `verilog_doc_parser` tool. The tool will generate `.rst` files
and a `verilog_index.rst` file that contains the `toctree` directive to include the generated
`.rst` files.

Contents
========

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   architecture/index
   verilog/index
   test/index



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
