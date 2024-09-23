.. _architecture:

Architecture
=============

This section provides an overview of the architecture of the OpenEye.

The rationale behind the OpenEye architecture is to provide a flexible and scalable platform for the development of high-performance computing systems for deep learning. 
The OpenEye architecture is based on a hierachical structure, with the following main components:

* A top-level control unit that is used to configure the system and to manage the execution of the application.
* PE Cluster: a cluster of Processing Elements (PEs) that are interconnected with each other.
* GLB Cluster: a cluster of Global Buffers (GLBs) that are used store intermediate data for reuse.
* Router Cluster: a cluster of routers that are used to connect adjacent PE and GLB clusters and to provide a connection to other router clusters.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    dataflow
    noc
    PE_cluster
    PE
    configuration




