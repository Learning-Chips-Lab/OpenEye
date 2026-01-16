# Hierarchical Mesh Network on Chip

## Introduction: Not a Traditional NoC

While commonly referred to as a Network-on-Chip (NoC) in the context of Eyeriss v2, the **Hierarchical Mesh Network (HM-NoC)** is technically not a conventional packet-switched NoC architecture. Traditional NoCs typically feature:
- Dynamic packet routing with headers
- Flow control mechanisms (virtual channels, buffers)
- Arbitration logic for competing traffic
- Runtime routing decisions

In contrast, Eyeriss v2's HM-NoC employs **circuit-switched routing** that is statically configured at layer mapping time, making it more accurately described as a **reconfigurable interconnection network** rather than a traditional NoC. Despite this distinction, the term "NoC" is used throughout the Eyeriss v2 literature for consistency with DNN accelerator terminology, and we maintain this convention here for the OpenEye, while acknowledging the architectural differences.

## Design Motivation

### The Challenge: Varying Data Reuse Across Layers

Modern compact and sparse DNNs present a fundamental challenge to on-chip network design:
- **Compact DNNs** (MobileNet, SqueezeNet, ResNet) have widely varying layer shapes and sizes
- **Different layer types** have drastically different data reuse patterns and bandwidth requirements
- **Data reuse characteristics** vary significantly across the three data types: input activations (iacts), weights, and partial sums (psums)

### Traditional NoC Limitations

Common NoC architectures used in DNN accelerators have rigid designs optimized for specific patterns:

| NoC Type | Strength | Weakness |
|----------|----------|----------|
| **Broadcast Network** | Maximizes data reuse | Low source bandwidth limits throughput when reuse is low |
| **Unicast Network** | High bandwidth | Cannot exploit data reuse opportunities |
| **All-to-All Network** | Perfect flexibility | Cost scales quadratically with number of nodes (impractical) |

### Example Failure Cases

**Spatial Accumulation Arrays** (weight-stationary):
- Rely on both output and input channels for parallelism
- Fail on depth-wise layers with few channels
- Result: Poor PE utilization

**Temporal Accumulation Arrays** (output-stationary):
- Optimized for specific dimension mappings
- Cannot adapt to varying layer shapes
- Result: Reduced performance and energy efficiency

The original Eyeriss used a multicast-based broadcast NoC that worked well for large CNNs but suffered from insufficient bandwidth when processing compact DNNs with limited data reuse, such as depth-wise layers in MobileNet.

## Hierarchical Mesh Architecture

### Two-Level Hierarchy

The HM-NoC solves the scalability problem of all-to-all networks through a **two-level hierarchical approach**:

```
Level 1 (Cluster Level): All-to-All Network
  - Each cluster contains 12 PEs (3×4 arrangement)
  - Full connectivity within cluster
  - Low cost due to small cluster size

Level 2 (Global Level): 2D Mesh Network
  - 8×2 cluster array (16 clusters total = 192 PEs)
  - Standard mesh topology connecting clusters
  - Cost scales linearly with number of clusters
```

### Key Architectural Features

**Cluster Structure:**
- Each cluster has 12 processing elements (3×4 PE array)
- All-to-all connectivity between any data source (GLB bank or mesh port) and any PE
- Three router instances per cluster (one per data type)

**Mesh Connectivity:**
- Standard 2D mesh connects 8×2 = 16 clusters
- Each router has 3 directional ports (optimized for 8×2 layout, omits east/west as needed)
- One port connects to local GLB cluster
- One port broadcasts to all 12 PEs in the cluster

**Implementation:**
- **Circuit-switched routing** using multiplexers
- **Static configuration** set by control logic during layer mapping
- **Minimal hardware cost** compared to packet-switched NoCs
- **Separate networks** for each data type (iacts, weights, psums)

## Operating Modes

The HM-NoC can be dynamically reconfigured into four distinct modes to match the data reuse and bandwidth requirements of different layer types:

### 1. High Bandwidth Mode (Unicast)

**Use Case:** Layers with minimal data reuse (e.g., fully-connected layers with small batch sizes)

**Operation:**
- Each GLB bank or off-chip I/O delivers unique data to different PEs
- Maximum source bandwidth utilization
- Each PE receives different data values

**Example:** Fully-connected layers where weights have little reuse across batch

**Configuration:**
```
GLB Bank 0 → PE[0,0]
GLB Bank 1 → PE[0,1]
GLB Bank 2 → PE[1,0]
...
Each PE gets unique data
```

### 2. High Reuse Mode (Broadcast)

**Use Case:** Layers with maximum data reuse (e.g., depth-wise convolutions with many output channels)

**Operation:**
- Single data source broadcasts to all PEs across all clusters
- Minimizes source bandwidth requirement
- All PEs receive the same data value

**Example:** Depth-wise layers where weights are reused across all spatial positions

**Configuration:**
```
GLB Bank 0 → All PEs in all clusters
Same data value delivered everywhere
```

### 3. Grouped Multicast Mode

**Use Case:** Layers with moderate, structured data reuse

**Operation:**
- PEs are divided into groups
- Each group receives the same data (multicast within group)
- Different groups receive different data
- Groups can span multiple clusters

**Example:** Standard convolution layers with moderate channel counts

**Configuration:**
```
Group 0: PE[0:3] receive data A
Group 1: PE[4:7] receive data B
Group 2: PE[8:11] receive data C
...
```

### 4. Interleaved Multicast Mode

**Use Case:** Layers requiring fine-grained, interleaved multicast patterns

**Operation:**
- Data is multicast in an interleaved pattern across PEs
- Alternating PEs receive the same data
- Enables complex mapping strategies

**Example:** Convolution layers where both iact and weight NoCs need multicast with different patterns

**Configuration:**
```
Data A → PE[0], PE[2], PE[4], ...
Data B → PE[1], PE[3], PE[5], ...
Interleaved distribution pattern
```

## Layer-Specific Examples

### Conventional Convolution Layers

**Characteristics:**
- High data reuse for both iacts and weights
- Multiple input/output channels
- Standard filter sizes (3×3, 5×5)

**NoC Configuration:**
- **Weight NoC:** Grouped multicast (weights reused across PEs processing same output channel)
- **Iact NoC:** Interleaved multicast (activations reused across PEs processing different output channels)
- **Result:** All PEs remain busy with minimum bandwidth

### Depth-Wise Convolution Layers

**Characteristics:**
- Very few input/output channels (often C_in = C_out = 1 per group)
- High weight reuse
- Minimal iact reuse

**NoC Configuration:**
- **Weight NoC:** Broadcast mode (same weights to all PEs)
- **Iact NoC:** High bandwidth/unicast mode (unique activations to each PE)
- **Result:** Exploits available weight reuse while providing necessary iact bandwidth

**Note:** This configuration addresses a major weakness of the original Eyeriss, which would underutilize PEs in these layers.

### Fully-Connected Layers

**Characteristics:**
- Little weight reuse (especially with batch size = 1)
- High iact reuse across different output neurons
- Large weight matrix

**NoC Configuration:**
- **Weight NoC:** High bandwidth/unicast mode (unique weights to each PE)
- **Iact NoC:** Broadcast mode (same activations to all PEs)
- **Result:** Opposite of depth-wise configuration, maximizes PE utilization for FC layers

## Implementation Details

### Router Architecture

Each mesh network router implements **circuit-switched routing** with the following components:

**Port Structure:**
- **4 source ports** (receive data): 3 from adjacent routers in mesh + 1 from local GLB
- **4 destination ports** (transmit data): 3 to adjacent routers + 1 to local PE cluster

**Handshaking Signals:**
- **Data (d):** Actual data values being transmitted
- **Enable (e):** Control signal indicating valid data
- **Ready (r):** Backpressure signal indicating receiver readiness

**Routing Logic:**
- **Mode configuration (m):** Static configuration bits determine routing mode
- **Enable generation:** Each source port generates 4 enable signals (one per destination)
- **Enable aggregation:** Destination ports OR together all source enables
- **Ready aggregation:** Source ports AND together all destination readys
- **Data multiplexing:** MUX selects data from enabled source port

**Routing Modes:**
- Unicast: Enable single destination port
- Horizontal multicast: Enable destinations in horizontal direction
- Vertical multicast: Enable destinations in vertical direction
- Broadcast: Enable all destination ports

### Data Type Specific Networks

Eyeriss v2 implements **three separate HM-NoCs**, one for each data type:

#### Input Activation (Iact) NoC

**Specifications:**
- **Port width:** 24 bits
- **Capacity:** 3 uncompressed 8-bit values OR 2 compressed 12-bit count-data pairs per cycle
- **Routers per cluster:** 3 (one per GLB SRAM bank)
- **Source ports:** 3 mesh directions + 1 GLB connection
- **Destination ports:** 3 mesh directions + 1 to all 12 PEs (via all-to-all network)

**Features:**
- Supports compressed sparse column (CSC) format
- Direct delivery to PE iact SPads
- Configurable for varying activation reuse patterns

#### Weight NoC

**Specifications:**
- **Port width:** 24 bits (same as iact)
- **Capacity:** 3 uncompressed 8-bit values OR 2 compressed 12-bit count-data pairs per cycle
- **Routers per cluster:** 3 (one per GLB SRAM bank)
- **Configuration:** Similar to iact NoC but with independent routing mode

**Features:**
- Supports CSC compressed weights
- Filter row distribution to PEs
- Optimized for weight stationary dataflow

#### Partial Sum (Psum) NoC

**Specifications:**
- **Port width:** 40 bits
- **Capacity:** 2 partial sums per cycle (20 bits each)
- **Routers per cluster:** 2 (fewer than iact/weight due to different bandwidth requirements)
- **Bidirectional:** Supports both distribution and accumulation

**Features:**
- Distributes initial psum values to PEs
- Collects and accumulates psums from PEs
- Reduced number of routers due to lower bandwidth requirements

### All-to-All Cluster Network

Within each cluster, the all-to-all network provides **complete flexibility**:

```
Connectivity:
- Any of 3 routers → Any of 12 PEs
- Implemented with multiplexer network
- Configuration bits select source for each PE
- Low cost due to small cluster size (12 PEs)
```

## Configuration and Control

### Static Configuration

The HM-NoC is configured **statically for each DNN layer**:

1. **Mapping phase:** Software determines optimal PE mapping for layer
2. **Mode selection:** Choose NoC modes for each data type based on:
   - Available data reuse
   - Required bandwidth
   - Layer dimensions
3. **Configuration loading:** Control logic sets configuration bits for all routers
4. **Execution:** NoC operates in configured mode for entire layer

### Configuration Bits

Each router stores configuration bits that determine:
- **Routing mode** (unicast, multicast patterns, broadcast)
- **Source-to-destination mappings**
- **Enable/disable for each port**

### Dynamic Reconfiguration

Between layers, the NoC can be **reconfigured**:
- No runtime overhead during layer execution
- Quick reconfiguration between layers (< 100 cycles typical)
- Supports heterogeneous layer sequences in modern DNNs

## Advantages Over Traditional NoCs

### 1. Scalability

**Traditional All-to-All NoC:**
- Cost: O(N²) where N = number of PEs
- Impractical for 192 PEs (36,864 connections)

**HM-NoC:**
- Cost: O(N) due to hierarchical structure
- Practical for 192 PEs (linear scaling with clusters)

### 2. Flexibility

**Fixed NoCs (Broadcast/Unicast):**
- Optimized for one pattern
- Poor performance on other patterns

**HM-NoC:**
- Adapts to different layer requirements
- Maintains high PE utilization across diverse DNNs

### 3. Energy Efficiency

**Packet-Switched NoC:**
- Dynamic routing overhead
- Buffer power consumption
- Arbitration logic

**HM-NoC:**
- Static routing (no packet headers)
- Minimal buffering
- Simple multiplexer-based implementation
- Lower power consumption

### 4. Implementation Cost

**Traditional NoC:**
- Complex arbitration
- Virtual channel buffers
- Routing computation logic

**HM-NoC:**
- Simple multiplexers
- Minimal control logic
- Configuration registers only

## Performance Impact

### Compact DNN Support

The HM-NoC enables Eyeriss v2 to efficiently process compact DNNs:

**MobileNet Performance:**
- Depth-wise layers: High bandwidth mode prevents stalls
- Point-wise layers: Multicast modes maintain high reuse
- Result: 12.6× speedup vs. original Eyeriss

### PE Utilization

Different architectures on depth-wise layers:

| Architecture | PE Utilization | Bottleneck |
|--------------|----------------|------------|
| Spatial Accumulation Array | 12.5% (2/16 PEs) | Insufficient output channels |
| Temporal Accumulation Array | 25% (4/16 PEs) | Insufficient input channels |
| Eyeriss (original) | 100% | Bandwidth limited by broadcast NoC |
| **Eyeriss v2 (HM-NoC)** | **100%** | **No bottleneck** |

### Bandwidth Adaptation

The HM-NoC provides **3-24× bandwidth scalability**:
- Broadcast mode: 1× bandwidth (maximum reuse)
- Grouped multicast: 2-8× bandwidth (moderate reuse)
- Unicast mode: 24× bandwidth (minimum reuse)

## Comparison with Other Interconnects

### vs. Bus-Based Architectures

**Bus:**
- Simple, low cost
- Severe bandwidth bottleneck
- Poor scalability

**HM-NoC:**
- Higher complexity but manageable
- Scalable bandwidth
- Supports 192 PEs efficiently

### vs. Crossbar

**Crossbar:**
- Full connectivity
- O(N²) cost
- Area and power prohibitive

**HM-NoC:**
- Partial connectivity (sufficient for DNN patterns)
- O(N) cost through hierarchy
- Practical for large PE arrays

### vs. Traditional Mesh NoC

**Traditional Mesh:**
- Packet switching
- Dynamic routing
- High flexibility but high overhead

**HM-NoC:**
- Circuit switching
- Static configuration
- Lower overhead, sufficient flexibility

## Design Trade-offs

### Benefits
✓ Scalable to large PE arrays (linear cost)
✓ Adapts to varying data reuse patterns
✓ Low implementation cost (mux-based)
✓ High bandwidth when needed
✓ Energy efficient (no dynamic routing)

### Limitations
✗ Requires compile-time layer analysis
✗ Not optimal for extremely irregular patterns
✗ Reconfiguration overhead between layers
✗ Limited to predetermined routing modes

### Justification

For DNN workloads, these trade-offs are highly favorable because:
- Layer parameters are known at compile time
- Data reuse patterns are predictable
- Reconfiguration overhead is amortized over layer execution
- Predetermined modes cover practical DNN layer types

## Summary

The Hierarchical Mesh Network in Eyeriss v2 represents a **pragmatic middle ground** between rigid, efficient interconnects and flexible, complex NoCs. While not a traditional packet-switched NoC, the HM-NoC provides:

- **Sufficient flexibility** to handle diverse DNN layer types (CONV, DW, FC)
- **Scalable architecture** through two-level hierarchy
- **Low implementation cost** via circuit switching
- **High performance** through bandwidth adaptation
- **Energy efficiency** by exploiting data reuse when available

This design philosophy—choosing "good enough" flexibility with minimal overhead—makes the HM-NoC a key enabler for Eyeriss v2's ability to efficiently execute compact and sparse DNNs on resource-constrained mobile devices.

## References

- Chen et al., "Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks on Mobile Devices", IEEE Journal of Solid-State Circuits, 2019
- The term "hierarchical mesh" emphasizes the architectural innovation while acknowledging that it's a specialized interconnection network rather than a traditional NoC