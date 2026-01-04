# Row Stationary Dataflow

## Overview

The **Row Stationary (RS)** dataflow is an energy-efficient data orchestration strategy designed for deep neural network (DNN) accelerators, originally introduced in Eyeriss and adopted in Eyeriss v2. It is adopted by OpenEye. RS dataflow specifically optimized to minimize data movement and maximize data reuse across the memory hierarchy by strategically keeping partial convolution results stationary within the processing elements (PEs).

## Motivation

Data movement is the dominant source of energy consumption in DNN accelerators, often consuming orders of magnitude more energy than the actual computation. The row stationary dataflow addresses this challenge by:

- **Minimizing off-chip memory accesses** through intelligent data reuse
- **Reducing inter-PE communication** by keeping intermediate results local
- **Maximizing exploitation of all types of data reuse** (convolutional, filter, input feature map)
- **Balancing bandwidth requirements** across different data types

## Core Principle

The fundamental idea behind row stationary dataflow is to **keep 1D convolution (row) of filter weights and corresponding input feature map (ifmap) activations stationary in each PE** to accumulate partial sums over multiple cycles. This approach enables:

1. **Convolutional reuse**: Input activations are reused across multiple filter weights
2. **Filter reuse**: Filter weights are reused across multiple input activations  
3. **Partial sum accumulation**: Reduces the need to move intermediate results

## Dataflow Operation

### Data Mapping

In the row stationary approach:

- Each **PE is assigned a specific 1D row of filter weights** (e.g., one row of a 3×3 filter)
- The corresponding **ifmap activations** that align with this filter row are also fetched to the PE
- **Partial sums (psums)** are accumulated locally within the PE across multiple clock cycles

### Processing Flow

```
For a convolution operation Y = W * X:
  
  PE[i,j] stores:
    - One row of filter weights W[row_i]
    - Streaming input activations X aligned to this filter row
    - Accumulating partial sum psums[i,j]
  
  Over time:
    1. Weights stay stationary in PE
    2. Input activations are streamed in as needed
    3. Partial sums accumulate with each MAC operation
    4. Final psums are sent to next stage when row processing completes
```

### Example: 3×3 Convolution

For a 3×3 filter convolution:

```
Filter W:           PE Array Mapping:
[w00 w01 w02]       PE0: [w00 w01 w02] ← Row 0
[w10 w11 w12]  -->  PE1: [w10 w11 w12] ← Row 1  
[w20 w21 w22]       PE2: [w20 w21 w22] ← Row 2

Each PE accumulates psums for its assigned filter row across the input feature map.
```

## Data Reuse Hierarchy

The row stationary dataflow exploits three levels of data reuse:

### 1. Convolutional Reuse (Highest Priority)
- Input activations are reused by **multiple filter weights** within the same PE
- Achieved by keeping the filter row stationary while streaming input activations

### 2. Filter Reuse
- Filter weights are reused across **multiple spatial locations** of the input feature map
- Weights remain in the PE across many input windows

### 3. Input Feature Map Reuse
- Input activations are reused across **multiple output channels**
- Achieved through strategic inter-PE data forwarding via the Network-on-Chip (NoC)

## Key Benefits

### Energy Efficiency
- **Reduced DRAM accesses**: Filter weights and many activations stay on-chip
- **Minimized inter-PE communication**: Partial sums accumulate locally
- **Optimized memory hierarchy usage**: Global buffer serves as staging area

### Flexibility
- Adapts to different layer shapes (varying filter sizes, channels, input dimensions)
- Supports different levels of parallelism based on available PE resources
- Compatible with sparse and compact neural networks

### Performance
- High PE utilization through balanced data distribution
- Overlapped computation and communication
- Scalable to different array sizes

## Implementation Considerations

### Memory Organization
- **PE Register Files**: Store stationary filter weights and accumulating partial sums
- **Global Buffer**: Shared scratchpad memory for staging ifmap activations and ofmap results
- **NoC**: Facilitates data distribution and collection between global buffer and PE array

### Mapping Strategy
- **Spatial mapping**: Distribute different filter rows across PE array
- **Temporal mapping**: Stream input activations over multiple cycles
- **Output mapping**: Collect and reduce partial sums to form final outputs

### Synchronization
- PEs operate in SIMD-like fashion for regular convolutions
- Flexible scheduling for layers with varying computational demands
- Support for both dense and sparse processing modes

## Comparison with Other Dataflows

| Dataflow | Primary Stationary Data | Optimal Use Case |
|----------|-------------------------|------------------|
| **Row Stationary** | Filter rows + Psums | Balanced reuse, energy efficiency |
| Weight Stationary | Filter weights | Large batch sizes, filter reuse |
| Output Stationary | Partial sums | Large filter dimensions |
| No Local Reuse | None (streaming) | Memory-constrained designs |

## Eyeriss v2 Enhancements

Eyeriss v2 extends the original row stationary concept with:

- **Hierarchical Mesh NoC**: Flexible interconnect adapting to different layer shapes
- **Compressed Processing**: Direct processing of sparse data in compressed format
- **Enhanced Data Gating**: Per-PE gating for zero-valued activations and weights
- **Flexible Mapping**: Support for diverse compact DNN architectures (MobileNet, SqueezeNet)

## Summary

The row stationary dataflow provides an elegant solution to the data movement challenge in DNN accelerators by strategically keeping filter rows and partial sums stationary within processing elements. This approach maximizes energy efficiency through exploiting multiple dimensions of data reuse while maintaining flexibility for diverse network architectures. For hardware designers and programmers, understanding this dataflow is essential for optimizing DNN workload mapping and achieving high performance and energy efficiency.

## References

- Chen et al., "Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks", ISSCC 2016
- Chen et al., "Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks on Mobile Devices", IEEE JSSC 2019