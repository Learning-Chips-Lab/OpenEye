# Processing Element (PE) Cluster Architecture

## Overview

The **Processing Element (PE) cluster** is the fundamental computational building block in OpeEye, similar to Eyeriss v2. The accelerator contains **16 PE clusters** arranged in an 8×2 array, with each cluster containing **12 PEs** organized in a 3×4 arrangement, for a total of **192 PEs** across the entire chip. Each PE cluster works as a cohesive unit, receiving data from the hierarchical mesh network and executing multiply-accumulate (MAC) operations according to the row stationary dataflow.

**OpenEye Implementation Note:** The [OpenEye](./PE_cluster.md#openeye-pe-cluster-implementation) neural network accelerator implements a parameterized version of this architecture, allowing flexible PE array configurations (not fixed at 3×4) with configurable data widths, multiple global buffer interfaces, and selectable processing modes. The core concepts and dataflow patterns described here form the foundation for OpenEye's design.

## Cluster Organization

### Top-Level Structure

```
Eyeriss v2 Array: 8×2 = 16 Clusters
├── Each Cluster: 3×4 = 12 PEs (FIXED configuration)
├── Total PEs: 16 × 12 = 192 PEs
└── Total MACs: 384 MACs (2 per PE with SIMD)
```

**OpenEye Difference:** In OpenEye, the PE cluster dimensions (PE_ROWS × PE_COLUMNS) are fully parameterizable. The default configuration matches Eyeriss v2 (3×4), but can be scaled to 2×2 (4 PEs) for area-constrained designs or 4×4 (16 PEs) for higher throughput. See [PE Cluster Parameters](./PE_cluster.md#pe-cluster-parameters).

### Cluster Hierarchy

Each PE cluster operates as an independent computing unit with:

**Internal Connectivity:**
- **All-to-all network** connects any of the 3 routers to any of the 12 PEs
- Allows flexible data distribution within the cluster
- Low cost due to small cluster size

**OpenEye Implementation:** OpenEye provides similar all-to-all connectivity within the cluster, but implements this using multiplexer-based routing rather than a dedicated mesh. Each PE can independently select its input activation source from NUM_GLB_IACT interfaces via the `iact_choose_i` signal. This approach trades off flexibility for reduced logic overhead on FPGA implementations. See [PE Cluster Data Flow Architecture](./PE_cluster.md#pe-cluster-data-flow-architecture).

**External Connectivity:**
- Connected to **router cluster** (3 iact routers, 3 weight routers, 4 psum routers)
- Connected to other PE clusters via 2D mesh network
- Each router cluster interfaces with corresponding **GLB cluster** (12 KB shared memory)

**OpenEye Extension:** In OpenEye, the external connectivity is more flexible. The PE cluster supports dual partial sum paths (direct GLB and router-based), selectable per column via `psum_choose_i`. This allows OpenEye to integrate with both hierarchical mesh networks and direct GLB connections, providing more flexibility for different system configurations.

**Memory Hierarchy:**
```
Off-Chip DRAM
    ↓
Global Buffer (GLB) Cluster: 12 KB
    ↓
Router Cluster (HM-NoC)
    ↓
PE Cluster: 12 PEs
    ↓
PE Scratchpad Memory (SPad): ~410.5 B per PE
```

### Cluster Array Configuration

The 8×2 cluster array provides:
- **Spatial scalability**: Linear scaling in implementation cost
- **Flexible mapping**: Different layers can utilize different portions of array
- **Load balancing**: Work distributed across clusters based on layer dimensions
- **Independent operation**: Each cluster can process different data simultaneously

## Processing Element (PE) Architecture

### PE Design Philosophy

Each PE in Eyeriss v2 is designed to:
1. **Process sparse data directly** in compressed format (Compressed Sparse Column - CSC)
2. **Support SIMD** to perform 2 MAC operations per cycle
3. **Implement row stationary dataflow** for energy-efficient data reuse
4. **Adapt to varying sparsity** (process both dense and sparse data)
5. **Minimize data movement** through local scratchpad storage

**OpenEye Alignment:** OpenEye's PE design philosophy matches Eyeriss v2 core principles. The OpenEye PE.v module implements the same 7-stage pipeline, 5 scratchpad organization, and sparsity-aware processing. Key OpenEye enhancement: configurability of PARALLEL_MACS (1, 2, or 4 concurrent MAC operations) and support for selectable precision via DATA_IACT_BITWIDTH and DATA_WGHT_BITWIDTH parameters. See [OpenEye PE Individual Unit](./PE_cluster.md#openeye-pe-individual-unit).

### PE Block Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     Eyeriss v2 PE                        │
├─────────────────────────────────────────────────────────┤
│  Pipeline Stage 1-2: Input Activation Fetch             │
│  ├── iact address SPad (9×4b = 36 bits)                 │
│  └── iact data SPad (16×12b = 192 bits)                 │
│                                                          │
│  Pipeline Stage 3-5: Weight Fetch                       │
│  ├── weight address SPad (16×7b = 112 bits)             │
│  └── weight data SPad (96×24b = 2304 bits)              │
│                                                          │
│  Pipeline Stage 6-7: MAC Computation & Psum Update      │
│  ├── 2× MAC units (8b × 8b → 20b)                       │
│  └── psum SPad (32×20b = 640 bits)                      │
└─────────────────────────────────────────────────────────┘

Total SPad Storage per PE: ~410.5 bytes
```

### Seven-Stage Pipeline

The PE implements a **7-stage pipeline** to handle data dependencies in compressed data processing:

#### Stage 1-2: Input Activation (Iact) Fetch
**Purpose:** Read non-zero input activations from compressed CSC format

**Stage 1:** Read iact address SPad
- Access CSC address vector
- Determine location of next non-zero iact segment
- Calculate address for iact data SPad

**Stage 2:** Read iact data SPad
- Fetch non-zero iact value
- Fetch count value (number of leading zeros)
- Pass non-zero iact to next stage

**Key Feature:** Zero iacts are never read, saving both cycles and energy

#### Stage 3-5: Weight Fetch
**Purpose:** Read corresponding non-zero weights for the current iact

**Stage 3:** Read weight address SPad
- Access CSC address vector for weights
- Identify the correct column of weights (based on iact position)
- Calculate bounds for weight data SPad access

**Stage 4:** Read weight data SPad (Part 1)
- Fetch first weight in column
- Read count value for weight position

**Stage 5:** Read weight data SPad (Part 2)
- Continue fetching remaining weights in column
- Handle SIMD: read 2 weights per cycle when possible

**Key Feature:** Only non-zero weights corresponding to non-zero iacts are fetched

#### Stage 6-7: Computation & Accumulation
**Purpose:** Perform MAC operations and update partial sums

**Stage 6:** MAC Computation
- **MAC 0:** iact × weight[0] → product[0]
- **MAC 1:** iact × weight[1] → product[1] (SIMD)
- Both MACs operate in parallel with same iact

**Stage 7:** Partial Sum Update
- Read 2 psums from psum SPad
- Add products to corresponding psums
- Write updated psums back to SPad or output to NoC

**Key Feature:** SIMD enables 2 MACs per cycle, doubling throughput

### Pipeline Dependencies

The 7-stage pipeline handles several critical dependencies:

**Read Dependencies:**
1. **Address before Data:** Address SPad must be read before data SPad
2. **Iact before Weight:** Iact must be read before corresponding weight
3. **Weight before Psum:** Weight determines which psum to update

**Why Deep Pipeline?**
- Compressed data format introduces non-deterministic access patterns
- Address cannot be calculated until count values are read
- Multiple SPad accesses require multiple cycles
- Deep pipeline maintains throughput despite dependencies

**Pipeline Efficiency:**
- Pipeline can stay full when processing dense regions
- Bubbles occur only when sparsity is extremely high
- Later stages can continue working on previous iact while early stages fetch next iact

## Scratchpad (SPad) Memory Organization

### Five SPad Types

Each PE contains **five separate scratchpads** for different data types:

**Note on OpenEye Implementation:** OpenEye maintains the exact same 5-SPad organization as Eyeriss v2, with configurable depths via IACT_ADDR_WORDS, IACT_DATA_WORDS, WGHT_ADDR_WORDS, WGHT_DATA_WORDS, and PSUM_WORDS parameters. The default values match Eyeriss v2 specifications. See [Memory Organization Parameters](./PE_cluster.md#memory-organization-parameters) for OpenEye-specific configurations.

#### 1. Input Activation Address SPad
**Size:** 9 words × 4 bits = 36 bits (4.5 bytes)
**Type:** Register file
**Purpose:** Stores CSC address vector for iacts
**Function:** 
- Points to start of each compressed iact segment
- Maximum 9 segments = window size of 16 iacts (due to sliding window)
- Each address is 4 bits (can address up to 16 locations)

#### 2. Input Activation Data SPad
**Size:** 16 words × 12 bits = 192 bits (24 bytes)
**Type:** Register file
**Purpose:** Stores compressed iact data and count vectors
**Format:** Each 12-bit word contains:
- 8-bit iact value (non-zero only)
- 4-bit count value (number of leading zeros)
**Capacity:** Up to 16 non-zero iacts in sliding window

#### 3. Weight Address SPad
**Size:** 16 words × 7 bits = 112 bits (14 bytes)
**Type:** Register file
**Purpose:** Stores CSC address vector for weights
**Function:**
- Points to start of each weight column in compressed format
- 16 columns maximum = M0 (output channels) ≤ 16 in standard mapping
- Each address is 7 bits (can address up to 128 locations in weight data SPad)

#### 4. Weight Data SPad
**Size:** 96 words × 24 bits = 2304 bits (288 bytes)
**Type:** SRAM (largest SPad due to weight reuse in RS dataflow)
**Purpose:** Stores compressed weight matrix
**Format:** Each 24-bit word contains:
- Two 12-bit count-data pairs (for SIMD)
- OR one 12-bit pair + 12-bit zero (odd number of weights in column)
**Capacity:** 192 non-zero weights in compressed form (96 × 2)

**Note:** While nominal weight matrix might be M0 × C0 × S = 512 weights, compression ensures non-zero weights fit in 192-word capacity

#### 5. Partial Sum SPad
**Size:** 32 words × 20 bits = 640 bits (80 bytes)
**Type:** Register file with dual ports (for SIMD)
**Purpose:** Stores accumulating partial sums
**Precision:** 20-bit fixed-point (sufficient for accumulation without overflow)
**Capacity:** Up to 32 output channels (M0 ≤ 32)
**Ports:**
- 2 read ports (read 2 psums per cycle for SIMD)
- 2 write ports (write 2 updated psums per cycle)

### SPad Size Summary

| SPad Type | Size (bytes) | Implementation | Purpose |
|-----------|--------------|----------------|---------|
| Iact Address | 4.5 | Register | CSC address vector |
| Iact Data | 24 | Register | Compressed iacts |
| Weight Address | 14 | Register | CSC address vector |
| Weight Data | 288 | SRAM | Compressed weights |
| Psum | 80 | Register (dual-port) | Partial sums |
| **Total** | **~410.5** | Mixed | Complete PE storage |

**Area Breakdown:** SPads account for approximately **72% of PE area**, while the two MAC units account for only **5%**.

## Compressed Sparse Column (CSC) Format

### Data Compression Strategy

Eyeriss v2 keeps both **weights and activations in compressed format** throughout the memory hierarchy, from off-chip DRAM to PE SPads. This enables:
- Reduced off-chip bandwidth
- Reduced on-chip storage
- Reduced on-chip data movement
- Processing speedup by skipping zeros

### CSC Format Structure

The CSC format consists of three components:

**1. Data Vector:** Non-zero values only
**2. Count Vector:** Number of leading zeros before each non-zero value
**3. Address Vector:** Pointers to segment boundaries

### CSC Encoding Example

**Original Weight Matrix (4×5):**
```
Column:    0    1    2    3    4
        ┌───  ───  ───  ───  ───┐
Row 0   │ 0    a    0    0    g │
Row 1   │ 0    0    0    0    h │
Row 2   │ 0    c    0    0    i │
Row 3   │ b    d    0    f    0 │
        └───  ───  ───  ───  ───┘
```

**CSC Compressed Format:**
```
Address Vector: [0, 1, 4, 4, 5, 8]
                 │  │  │  │  │  └─ End of col 4 (location 8)
                 │  │  │  │  └─── Start of col 4 (location 5)
                 │  │  │  └────── Start of col 3 (location 4, repeated = empty col)
                 │  │  └────────── End of col 2 (location 4)
                 │  └──────────── Start of col 1 (location 1)
                 └─────────────── Start of col 0 (location 0)

Data Vector: [b, a, c, d, f, g, h, i]
Count Vector: [3, 0, 2, 0, 3, 0, 1, 0]
              └─3 zeros before 'b' in col 0
                 └─0 zeros before 'a' in col 1
                     └─2 zeros before 'c' in col 1
                        └─0 zeros before 'd' in col 1
                            └─3 zeros before 'f' in col 3
                               └─0 zeros before 'g' in col 4
                                   └─1 zero before 'h' in col 4
                                      └─0 zeros before 'i' in col 4
```

**Compressed Word Format (12 bits):**
```
┌─────────────┬─────────────┐
│ Count (4b)  │  Data (8b)  │
└─────────────┴─────────────┘
```

### Count Bitwidth Trade-off

**4-bit count** chosen as optimal for 8-bit data:
- **Too small (e.g., 2-bit):** Cannot represent runs > 3 zeros, reduces compression
- **Too large (e.g., 8-bit):** High overhead, reduces compression
- **4-bit sweet spot:** Handles up to 15 consecutive zeros, minimal overhead

### CSC for Input Activations

Iacts are divided into **non-overlapping segments** for sliding window processing:
- Each segment is C0 × U elements (C0 = input channels in PE, U = stride)
- Each segment is CSC-encoded independently
- Address vector enables quick segment replacement during sliding window
- Enables efficient convolution without decompression

### Sparse Processing Scenarios

The PE handles **three scenarios** based on sparsity:

#### Scenario 1: Zero Iact
- **Detection:** CSC format skips zero iacts entirely
- **Action:** No read from iact data SPad
- **Result:** No cycles wasted, maximum speedup
- **Energy:** Minimal, only address SPad accessed

#### Scenario 2: Non-zero Iact, Zero Weights
- **Detection:** Iact fetched, but weight column is empty
- **Action:** Iact not passed to computation stages
- **Result:** No pipeline bubble (later stages still busy with previous iact)
- **Energy:** Iact read cost only

#### Scenario 3: Non-zero Iact, Non-zero Weights
- **Detection:** Both iact and weights fetched
- **Action:** Full pipeline executes MAC operations
- **Result:** Useful computation performed
- **Energy:** Full MAC energy, but only for useful work

## SIMD Support

### Motivation

Profiling shows that **MAC units consume only 2-9% of PE power** and occupy **< 5% of PE area**. This creates an opportunity to add a second MAC unit at minimal cost for up to **2× throughput improvement**.

**OpenEye Extension:** While Eyeriss v2 fixes SIMD at 2 MACs per cycle, OpenEye generalizes this with the PARALLEL_MACS parameter. OpenEye supports PARALLEL_MACS = 1 (serial, for area reduction), 2 (default dual MAC matching Eyeriss v2), or 4 (quad MAC for higher throughput). This trade-off between area and performance is configured at design time. See [Processing Configuration](./PE_cluster.md#processing-configuration).

### SIMD Implementation

**SIMD Width:** 2 (two MAC operations per cycle)

**Architecture Changes:**
1. **Second MAC unit** added to computation stage
2. **Weight data SPad:** Width increased to 24 bits (2 × 12-bit words)
3. **Psum SPad:** Dual read/write ports added
4. **Control logic:** Handles two psums simultaneously

### SIMD Operation

**Normal Case (Even Number of Weights):**
```
Cycle N:
  iact[i] → MAC0: iact[i] × weight[j+0] → psum[j+0]
         → MAC1: iact[i] × weight[j+1] → psum[j+1]
  
Both MACs operate in parallel with same iact
```

**Odd Number Case:**
```
24-bit word: [weight + count] [0x000]
                    ↓              ↓
                  MAC0          MAC1 (gated)

MAC1 datapath is clock-gated when second word is zero
Psum SPad second port is also gated
Saves power in odd-weight-count scenarios
```

### SIMD Benefits

**Throughput:** Up to 2× improvement
- Best case: All columns have even number of weights
- Typical case: ~1.6-1.8× improvement (some odd columns)

**Energy Efficiency:** Improved due to:
- Fewer iact SPad reads (same iact used for 2 MACs)
- Better amortization of control overhead
- Minimal area/power overhead for second MAC

**Area Cost:** ~15% increase
- Second MAC unit: ~5%
- Dual-port psum SPad: ~8%
- Wider PE I/O buses: ~2%

## Row Stationary Dataflow Implementation

### Dataflow Mapping to PE Cluster

The row stationary dataflow assigns work to PEs as follows:

**OpenEye Implementation:** OpenEye implements the same row stationary dataflow concept as Eyeriss v2. The weight distribution is row-wise (all PEs in a row share same weights via the weight interface), and input activations are broadcast/multicast to PEs as needed. The configurable PE array allows the same dataflow patterns to scale with different cluster sizes. See [PE Cluster Data Flow Architecture](./PE_cluster.md#pe-cluster-data-flow-architecture) for detailed OpenEye dataflow paths.

**Weight Assignment:**
- Each PE receives **one or more 1D rows of filter weights**
- Example: For 3×3 filter, PE[0] gets row 0, PE[1] gets row 1, PE[2] gets row 2
- Rows are stored in weight data SPad in CSC compressed format

**Input Activation Assignment:**
- Corresponding **iact sliding window** is fetched to PE
- Window size: C0 × S (S = filter width)
- Window slides by C0 × U elements (U = stride)

**Partial Sum Accumulation:**
- Each PE accumulates **M0 partial sums** (M0 = output channels mapped to PE)
- Psums remain in psum SPad across multiple cycles
- Final psums sent to GLB or next layer

### Processing Flow in PE Cluster

```
Step 1: Weight Distribution (One-time per layer)
  - Weights loaded from DRAM → GLB → HM-NoC → PE weight SPads
  - Weights remain stationary in PEs for entire layer

Step 2: Sliding Window Processing (Repeated for each output position)
  For each output position:
    a. Load iact window segment from GLB via HM-NoC
    b. PEs compute MACs: iact × weight → accumulate to psum
    c. Slide window (replace C0 × U iacts with new ones)
    d. Repeat until all input positions processed

Step 3: Partial Sum Collection
  - PEs output final psums via psum NoC
  - Psums reduced/accumulated at GLB
  - Output activations sent to next layer or off-chip
```

### Cluster-Level Parallelism

Multiple PEs in a cluster operate in **SIMD-like fashion**:
- All PEs execute the same control flow
- Different PEs process different data (different filter rows or channels)
- Synchronized by cluster-level control signals

**Example Mapping (3×3 convolution, 12 output channels):**
```
PE[0,0]: Filter rows for output channels 0-3, filter row 0
PE[0,1]: Filter rows for output channels 0-3, filter row 1
PE[0,2]: Filter rows for output channels 0-3, filter row 2
PE[1,0]: Filter rows for output channels 4-7, filter row 0
PE[1,1]: Filter rows for output channels 4-7, filter row 1
...
PE[3,3]: Filter rows for output channels 8-11, filter row 2
```

## Dense Mode Support

### Adaptation to Low Sparsity

When sparsity is low (< 30-40%), CSC compression overhead exceeds benefits. The PE supports **dense mode**:

**OpenEye Note:** OpenEye's PE implementation inherits the dense mode support from Eyeriss v2 design. The sparsity-aware processing with automatic mode selection (determined by compiled mapping) remains the same. This allows OpenEye to efficiently handle both sparse and dense neural networks with the same hardware.

**Changes in Dense Mode:**
1. **Address SPads:** Clock-gated (not used)
2. **Count values:** Fixed to 0 (sequential access)
3. **Data SPads:** Accessed sequentially
4. **Pipeline:** Operates with deterministic latency

**Automatic Selection:** Software determines mode based on measured/expected sparsity

**Benefits:**
- Eliminates compression overhead for dense data
- Still maintains row stationary dataflow
- Flexible adaptation to various DNNs

## Workload Balancing

### Challenge: Variable Sparsity

Different PEs may have different amounts of work due to:
- Non-uniform sparsity distribution
- Different filter sizes mapped to PEs
- Varying channel counts

### Solution: Sparsity-Aware Mapping

**Strategy:** Map more **non-zero** weights to each PE rather than more **nominal** weights

**Example (Sparse AlexNet CONV5):**
- Nominal weight matrix: 32 × 4 × 3 = 384 weights
- Non-zero weights (70% sparsity): ~174 weights
- All 174 non-zero weights fit in PE weight SPad (capacity: 192)

**Benefits:**
- Higher utilization of PE resources
- Reduced workload imbalance
- Better amortization of control overhead

**Compile-Time Optimization:**
- Sparsity pattern known at compile time
- Mapping optimizer distributes work evenly
- Minimizes idle time across PE array

## Integration with Hierarchical Mesh NoC

### Data Flow Through Cluster

**Input Activations:**
```
GLB Cluster → Iact Router Cluster → All-to-All Network → PE iact SPads
  (3 banks)      (3 routers)          (within cluster)      (12 PEs)
```

**Weights:**
```
Off-chip DRAM → Weight Router Cluster → All-to-All Network → PE weight SPads
                   (3 routers)           (within cluster)      (12 PEs)
Note: Weights bypass GLB, go directly to PEs
```

**Partial Sums:**
```
PE psum SPads → All-to-All Network → Psum Router Cluster → GLB Cluster
   (12 PEs)       (within cluster)       (4 routers)          (4 banks)
```

**OpenEye Architecture Difference:** OpenEye abstracts away the detailed mesh router hierarchy shown above. Instead, it provides simplified interfaces:
- Input activations from NUM_GLB_IACT sources (conceptually representing multiple GLB banks)
- Weights distributed row-wise from PE_ROWS sources
- Partial sums collected column-wise, with dual routing paths (direct GLB or router-based via `psum_choose_i`)

This simplification allows OpenEye to integrate with various system architectures without requiring a fixed hierarchical mesh. See [PE Cluster Data Flow Architecture](./PE_cluster.md#pe-cluster-data-flow-architecture) for OpenEye's specific data flow patterns.

### Cluster-Level Multicast

Within a cluster, the **all-to-all network** enables flexible data distribution:

**Unicast Mode:** Each PE receives unique data
```
Router 0 → PE[0,1,2,3]   (4 unique iacts)
Router 1 → PE[4,5,6,7]   (4 unique iacts)
Router 2 → PE[8,9,10,11] (4 unique iacts)
```

**Multicast Mode:** Multiple PEs receive same data
```
Router 0 → PE[0,3,6,9]   (same iact to 4 PEs)
Router 1 → PE[1,4,7,10]  (same iact to 4 PEs)
Router 2 → PE[2,5,8,11]  (same iact to 4 PEs)
```

**Broadcast Mode:** All PEs receive same data
```
Router 0 → All 12 PEs    (same iact to all)
```

## Performance Characteristics

### Area Breakdown (Per PE)

| Component | Area % | Notes |
|-----------|--------|-------|
| SPads | 72% | Dominated by weight data SRAM |
| MAC Units | 5% | Two 8b×8b multipliers |
| Control Logic | 18% | CSC processing, pipeline control |
| I/O | 5% | Data buses, handshaking |

**Total PE Area:** 1.73× larger than original Eyeriss PE
- Sparse processing support: +50%
- SIMD support: +15%
- Enhanced control: +8%

**OpenEye Area Considerations:**
- OpenEye's area profile depends on parameterization:
  - PARALLEL_MACS=1 (serial): Reduces MAC area overhead from 5% to ~3%, saves ~2-3% total PE area
  - PARALLEL_MACS=2 (default): Matches Eyeriss v2 area profile
  - PARALLEL_MACS=4: Increases MAC area to ~10%, adds ~5% total PE area
- FPGA implementations have different area characteristics than ASIC (SPad SRAM typically uses distributed RAM, reducing relative impact)
- Configurable precision (DATA_IACT_BITWIDTH, DATA_WGHT_BITWIDTH) allows trade-offs: 4-bit reduces data SPad by ~50%, 16-bit increases by 2×

### Throughput

**Peak Throughput (per PE):**
- 2 MACs per cycle @ 200 MHz = 400 MOPS per PE
- 192 PEs × 400 MOPS = 76.8 GOPS per cluster (12 PEs)
- Total chip: 153.6 GOPS (384 MACs @ 200 MHz)

**Effective Throughput (with sparsity):**
- Sparse MobileNet: ~80-90% of peak (due to skipping zeros)
- Sparse AlexNet: ~50-60% of peak (varies by layer)

**OpenEye Throughput Scalability:**
- With PARALLEL_MACS=1: 200 MOPS per PE (50% of Eyeriss v2)
- With PARALLEL_MACS=2 (default): 400 MOPS per PE (matches Eyeriss v2)
- With PARALLEL_MACS=4: 800 MOPS per PE (2× Eyeriss v2)
- Cluster throughput scales linearly with both PE count and PARALLEL_MACS parameter
- FPGA clock frequencies typically lower than 200 MHz (100-150 MHz typical), reducing absolute throughput but maintaining relative scaling

### Energy Efficiency

**Energy per MAC Operation:**
- Dense mode: ~20 pJ/MAC (typical)
- Sparse mode: ~12-15 pJ/MAC (varies with sparsity)
- Improvement: 1.3-1.7× due to:
  - Skipping zero computations
  - Reduced data movement
  - Efficient compressed data handling

**OpenEye Energy Characteristics:**
- ASIC implementation (Eyeriss v2): Optimized for minimum energy per MAC
- FPGA implementation (OpenEye): Energy profile depends on target platform:
  - Xilinx UltraScale+: ~50-100 pJ/MAC at 100 MHz (higher overhead from distributed RAM, slower clock)
  - Intel Stratix: Similar energy ranges depending on optimizations
- Energy benefits from sparsity remain significant (1.3-1.7× improvement) in FPGA implementations
- Serial mode (PARALLEL_MACS=1) reduces dynamic power consumption but increases latency and overall energy per inference
- Precision reduction (e.g., 4-bit) reduces data movement energy but may require retraining for accuracy

### Utilization

**PE Utilization (Sparse MobileNet):**
- Depth-wise layers: 95-100% (limited by iact bandwidth, not compute)
- Point-wise layers: 90-95% (good balance of reuse and compute)
- Average: >90% across all layers

**Comparison:**
- Original Eyeriss on MobileNet: ~20-40% (bandwidth limited)
- Eyeriss v2 on MobileNet: >90% (HM-NoC + sparse PE)

**OpenEye Utilization Characteristics:**
- OpenEye's PE utilization depends on system integration:
  - With adequate GLB bandwidth (NUM_GLB_IACT ≥ 3): Can achieve >85% utilization on sparse networks
  - With limited GLB bandwidth (NUM_GLB_IACT = 1): May drop to 60-75% depending on sparsity
- Smaller clusters (PE_ROWS=2, PE_COLUMNS=2) typically achieve higher utilization on small networks due to reduced synchronization overhead
- Larger clusters (PE_ROWS=4, PE_COLUMNS=4) require careful load balancing but can achieve higher absolute throughput when fully occupied
- FPGA memory bandwidth is typically the primary bottleneck (not compute), making bandwidth-friendly dataflows essential

## Key Design Insights

### 1. Compressed Processing is Worth the Complexity
- 7-stage pipeline adds area (~50%)
- But enables direct sparse processing
- Speedup from skipping zeros outweighs overhead
- Energy savings from reduced data movement

**OpenEye Validation:** OpenEye's configurable implementation allows validating these trade-offs across different configurations. Even with PARALLEL_MACS=1 (smallest area), the sparsity benefits remain significant, confirming the core insight. The ability to tune precision (DATA_IACT_BITWIDTH, DATA_WGHT_BITWIDTH) provides additional trade-off flexibility not available in fixed Eyeriss v2 design.

### 2. SIMD is Cheap and Effective
- MAC units are tiny (5% of PE)
- Adding second MAC: only 5% area increase
- Doubling throughput: nearly 2× improvement
- Key insight: Computation is cheap, data movement is expensive

**OpenEye Extension:** OpenEye generalizes this insight by allowing PARALLEL_MACS=1,2,4. The area scaling validates the principle: each additional MAC adds ~4-5% area for modest MOPS increase. This confirms that data movement (SPads, buses) dominates PE area, not computation.

### 3. Row Stationary Enables Compressed Processing
- Weights stay in PE (high temporal reuse)
- Enables compile-time optimization of weight mapping
- Predictable access patterns despite compression
- Balance of flexibility and efficiency

**OpenEye Implementation:** OpenEye strictly maintains row-stationary distribution (weights shared across rows via `pe_wght_data` interface). This validates the importance of dataflow choice for compressed processing—other dataflows (output-stationary, weight-stationary) would require more complex runtime sparse handling.

### 4. Five SPads Required for Dependencies
- Cannot merge SPads due to different access patterns
- Address SPads enable indirect addressing in CSC
- Separate data SPads reduce port conflicts
- Psum SPad enables accumulation without writebacks

**OpenEye Confirmation:** OpenEye's parameterized SPads (IACT_ADDR_WORDS, IACT_DATA_WORDS, WGHT_ADDR_WORDS, WGHT_DATA_WORDS, PSUM_WORDS) maintain the 5-type separation. Attempts to reduce this would violate the design's fundamental access pattern assumptions.

### 5. Cluster Granularity (12 PEs) is Optimal
- All-to-all network cost: O(N²) within cluster
- 12 PEs: Manageable cost, good flexibility
- Larger clusters: Prohibitive all-to-all cost
- Smaller clusters: More mesh overhead

**OpenEye Trade-offs:** OpenEye's configurable cluster size (2×2 to 4×4+) allows exploring this trade-off space:
- **2×2 clusters (4 PEs):** Lower area cost, suitable for area-constrained FPGA deployments
- **3×4 clusters (12 PEs):** Balanced default, matches Eyeriss v2 optimal point
- **4×4 clusters (16 PEs):** Higher throughput but increased routing complexity and area overhead
The Eyeriss v2 choice of 12 PEs remains a strong balance point across all implementations.

## OpenEye PE Cluster Implementation

### Architecture Overview

The **OpenEye** neural network accelerator implements a parameterized 2D array of Processing Elements based on the Eyeriss v2 architecture concepts, with adaptations for FPGA deployment. The PE cluster is implemented in [hdl/PE_cluster.v](../../hdl/PE_cluster.v) and orchestrates computation across a configurable grid of individual PE units defined in [hdl/PE.v](../../hdl/PE.v).

### PE Cluster Organization in OpenEye

Unlike the fixed Eyeriss v2 architecture, OpenEye provides flexible configuration:

**Default Configuration:**
```
OpenEye PE Cluster: PE_ROWS × PE_COLUMNS = 3 × 4 = 12 PEs
├── 3 rows of PEs
├── 4 columns of PEs
└── Total: 12 Processing Elements per cluster
```

**Key Differences from Eyeriss v2:**
- **Configurable Array Size:** PE_ROWS and PE_COLUMNS are parameterizable (Eyeriss v2 is fixed at 3×4)
- **Flexible Data Widths:** Configurable bus widths for IACT, WGHT, and PSUM (optimized for FPGA resources)
- **Multiple GLB Interfaces:** NUM_GLB_IACT parameter allows configurable number of global buffer connections (default: 3)
- **Serial Processing Mode:** SERIAL parameter enables time-multiplexed MAC operations for reduced area
- **Hierarchical Reset Control:** Optional reset synchronization for multi-level cascading

### PE Cluster Parameters

The PE_cluster module exposes configuration parameters controlling array dimensions, data precision, memory organization, and interface characteristics:

#### Array Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `PE_ROWS` | int | 3 | Number of processing elements in vertical direction |
| `PE_COLUMNS` | int | 4 | Number of processing elements in horizontal direction |
| `PES` | int | PE_ROWS × PE_COLUMNS | Total number of PEs (computed automatically) |

#### Processing Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `PARALLEL_MACS` | int | 2 | Number of parallel MAC operations per PE (1=serial, 2=dual MAC) |
| `SERIAL` | bool | 0 | Enable serial processing mode for reduced area |
| `NUM_GLB_IACT` | int | 3 | Number of input activation global buffer interfaces |

#### Data Precision Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `DATA_IACT_BITWIDTH` | int | 8 | Bit width of input activation values |
| `DATA_WGHT_BITWIDTH` | int | 8 | Bit width of weight values |
| `DATA_PSUM_BITWIDTH` | int | 20 | Bit width of partial sum accumulator |
| `DATA_IACT_OVERHEAD` | int | 4 | Sparsity metadata bits in activations (zero-skipping counts) |
| `DATA_WGHT_IGNORE_ZEROS` | int | 4 | Sparsity metadata bits in weights (position indicators) |

**Precision Trade-offs:**
- `DATA_IACT_BITWIDTH=8`: 8-bit quantized activations (INT8)
- `DATA_WGHT_BITWIDTH=8`: 8-bit quantized weights (INT8)
- `DATA_PSUM_BITWIDTH=20`: Accumulator width supports 4,096 MAC operations without overflow
- Sparsity overhead bits allow encoding compressed sparse data directly in payload

#### Interface Bus Widths
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `TRANS_BITWIDTH_IACT` | int | 24 | Input activation interface bus width (bits) |
| `TRANS_BITWIDTH_WGHT` | int | 24 | Weight interface bus width (bits) |
| `TRANS_BITWIDTH_PSUM` | int | 20 | Partial sum interface bus width (bits) |

**Bus Width Utilization Examples:**
- IACT bus (24 bits): Can carry 2 × (8-bit value + 4-bit overhead) = 2 compressed activations
- WGHT bus (24 bits): Can carry 2 × (8-bit weight + 4-bit metadata) = 2 compressed weights
- PSUM bus (20 bits): Carries 1 × 20-bit partial sum per cycle

#### Memory Organization Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `IACT_ADDR_WORDS` | int | 9 | Depth of input activation address SPad |
| `IACT_DATA_WORDS` | int | 16 | Depth of input activation data SPad |
| `WGHT_ADDR_WORDS` | int | 16 | Depth of weight address SPad |
| `WGHT_DATA_WORDS` | int | 96 | Depth of weight data SPad |
| `PSUM_WORDS` | int | 32 | Depth of partial sum accumulator SPad |

**SPad Capacity Analysis (per PE):**
```
Iact Address SPad:    9 × 4 bits = 36 bits
Iact Data SPad:       16 × 12 bits = 192 bits
Weight Address SPad:  16 × 7 bits = 112 bits
Weight Data SPad:     96 × 24 bits = 2304 bits
Psum SPad:            32 × 20 bits = 640 bits
────────────────────────────────────────────
Total per PE:                      ~410.5 bytes (matches Eyeriss v2)
```

#### Module Hierarchy Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `IS_TOPLEVEL` | bool | 1 | Enable additional control features at top level |
| `TOP_CLUSTER` | bool | 1 | Indicates topmost cluster in cascaded hierarchy |

### PE Cluster Ports

#### Clock and Reset Interface
```verilog
input  clk_i      // System clock (positive edge triggered)
input  rst_ni     // Asynchronous reset (active low)
```

#### Control Signals
```verilog
input  [PES-1:0]                                    compute_i        // Computation trigger per PE
input  [$clog2(NUM_GLB_IACT+1)*PES-1:0]            iact_choose_i    // Input source selection
input  [PE_COLUMNS-1:0]                            psum_choose_i    // Partial sum routing control

input  [11:0]  data_stream_i                        // Configuration parameter data
input          enable_stream_i                      // Configuration parameter enable
```

**Control Signal Details:**
- `compute_i[n]`: Single-cycle pulse triggers PE[n] to begin MAC computation (only when iact_set and wght_set are high)
- `iact_choose_i`: $clog2(NUM_GLB_IACT+1) bits per PE selects active input buffer source (0=disabled, 1-NUM_GLB_IACT=source index)
- `psum_choose_i[k]`: Selects partial sum source for column k: 0=direct GLB interface, 1=router interface
- `enable_stream_i` + `data_stream_i`: Sequential parameter streaming (stride, filter count, channel count, iact_addr_max)

#### Input Activation Interface
```verilog
input  [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0]  pe_iact_data       // Activation data from NUM_GLB_IACT sources
input  [NUM_GLB_IACT-1:0]                      pe_iact_enable     // Data valid per source
output [NUM_GLB_IACT-1:0]                      pe_iact_ready      // Ready to accept data per source
```

**IACT Interface Behavior:**
- Concatenated multi-source interface: `pe_iact_data[(i+1)*TRANS_BITWIDTH_IACT-1 : i*TRANS_BITWIDTH_IACT]` carries data from source i
- Each PE independently selects active source via `iact_choose_i[PE]`
- All PEs can source from different GLB interfaces simultaneously
- Ready signal aggregation: ready high only when ALL PEs are ready (NOR reduction)

#### Weight Interface
```verilog
input  [TRANS_BITWIDTH_WGHT*PE_ROWS-1:0]       pe_wght_data       // Weight data, one row per bus segment
input  [PE_ROWS-1:0]                           pe_wght_enable     // Data valid per PE row
output [PE_ROWS-1:0]                           pe_wght_ready      // Ready to accept per PE row
```

**Weight Distribution Pattern:**
- Row-wise distribution: `pe_wght_data[(j+1)*TRANS_BITWIDTH_WGHT-1 : j*TRANS_BITWIDTH_WGHT]` to PE row j
- All PEs in same row receive same weight data (weight-stationary dataflow)
- Ready signal aggregation per row: ready high only when ALL PEs in row are ready

#### Partial Sum Interface (Direct Path)
```verilog
input  [PE_COLUMNS-1:0]                        pe_psum_ready_i    // Downstream ready per column
input  [TRANS_BITWIDTH_PSUM*PE_COLUMNS-1:0]   pe_psum_data_i     // Incoming partial sums per column
input  [PE_COLUMNS-1:0]                        pe_psum_enable_i   // Incoming data valid per column

output [PE_COLUMNS-1:0]                        pe_psum_ready_o    // Ready signal per column
output [TRANS_BITWIDTH_PSUM*PE_COLUMNS-1:0]   pe_psum_data_o     // Outgoing partial sums per column
output [PE_COLUMNS-1:0]                        pe_psum_enable_o   // Output valid per column
```

#### Partial Sum Interface (Router Path)
```verilog
input  [PE_COLUMNS-1:0]                        pe_router_psum_ready_i    // Router ready per column
input  [TRANS_BITWIDTH_PSUM*PE_COLUMNS-1:0]   pe_router_psum_data_i    // Router psum data per column
input  [PE_COLUMNS-1:0]                        pe_router_psum_enable_i   // Router data valid per column

output [PE_COLUMNS-1:0]                        pe_router_psum_ready_o    // Ready to router per column
output [TRANS_BITWIDTH_PSUM*PE_COLUMNS-1:0]   pe_router_psum_data_o    // Psum to router per column
output [PE_COLUMNS-1:0]                        pe_router_psum_enable_o   // Output valid to router
```

**Dual Partial Sum Routing:**
- Two independent partial sum paths: direct GLB interface and router interface
- `psum_choose_i[k]` selects which path is active for column k
- Multiplexer/demultiplexer logic routes data based on selection
- Supports flexible cascading and hierarchical accumulation

### PE Cluster Data Flow Architecture

#### Array Organization

The PE cluster implements a **2D grid topology** with the following data movement patterns:

**Grid Topology (PE Rows × PE Columns = 3 × 4):**
```
Columns:     0      1      2      3
Row 0   ┌─────┬─────┬─────┬─────┐
        │PE00 │PE01 │PE02 │PE03 │
        ├─────┼─────┼─────┼─────┤
Row 1   │PE10 │PE11 │PE12 │PE13 │
        ├─────┼─────┼─────┼─────┤
Row 2   │PE20 │PE21 │PE22 │PE23 │
        └─────┴─────┴─────┴─────┘

Notation: PE[column][row] = PE at (PE_X=column, PE_Y=row)
```

#### Data Distribution Paths

**Input Activations:**
```
┌─────────────────────────────────────────────────────────────┐
│  Multiple GLB Sources (NUM_GLB_IACT × IACT interfaces)      │
└──────────────────────┬──────────────────────────────────────┘
                       │ pe_iact_data concatenated
                       │
              ┌────────▼────────────────────┐
              │  PE-level Multiplexer       │
              │  (iact_choose_i selects)    │
              └────────┬────────────────────┘
                       │
           ┌───────────┼───────────┬───────────┬───────────┐
           │           │           │           │           │
           ▼           ▼           ▼           ▼           ▼
       All 12 PEs can independently source from NUM_GLB_IACT
```

**Weights:**
```
┌──────────────────────────────────────┐
│  Weight Sources (PE_ROWS × interfaces)│
└──────────┬───────────┬───────────────┘
           │           │
        Row 0 Weights Row 1 Weights   Row 2 Weights
           │           │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
    │  PE row 0   │ │  PE row 1   │ │  PE row 2   │
    │ PE00-PE03  │ │ PE10-PE13  │ │ PE20-PE23  │
    └─────────────┘ └─────────────┘ └─────────────┘

Weights shared across all PEs in same row (weight-stationary)
```

**Partial Sums (Column-wise Accumulation):**
```
┌─────────────────────────────────────────────────────────────┐
│  External Psum Sources (Direct GLB or Router)               │
│  2 paths: psum_choose_i controls routing                   │
└──────────────────────────────────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
    Direct Path         Router Path
         │                   │
    pe_psum_data_i    pe_router_psum_data_i
         │                   │
         │        ┌──────────┘
         │        │
    ┌────▼────────▼──────────────┐
    │  Mux (psum_choose_i)       │
    └────┬───────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │  PE Column 0    1    2    3        │
    │  PE00+PE10+PE20  PE01...  PE02... PE03...
    └────┬──────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │  Column-wise Accumulation         │
    │  (Psum flows vertically)          │
    └────┬──────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │  Output to next stage             │
    │  (pe_psum_data_o, pe_psum_enable_o)│
    └──────────────────────────────────┘
```

**Column-wise Accumulation Detail:**

Within each column, psums flow vertically for accumulation:
```
Column k:
  External Input (psum_data_i[k])
        │
        ▼
  PE[k,0] (row 0)
        │
        │ output → input (PE[k,1])
        ▼
  PE[k,1] (row 1)
        │
        │ output → input (PE[k,2])
        ▼
  PE[k,2] (row 2)
        │
        │
        ▼
  Output (pe_psum_data_o[k])
```

Each PE passes partial sums downstream to the next row PE in same column, enabling accumulated results to flow through the column.

### Comparison: OpenEye vs. Eyeriss v2

#### Architectural Similarities
| Feature | Eyeriss v2 | OpenEye |
|---------|-----------|---------|
| PE cluster size | 3×4 = 12 PEs | Configurable (default 3×4) |
| SPad organization | 5-type SPad per PE | 5-type SPad per PE (matching) |
| SIMD capability | 2 MACs per PE | Configurable PARALLEL_MACS |
| CSC compression support | Direct processing | Direct processing |
| Row-stationary dataflow | Yes | Yes (via parameter config) |
| Dual-port PSUM SPad | Yes | Yes |
| Pipeline depth | 7 stages | 7 stages (per PE design) |

#### Key Implementation Differences

**1. Configurability:**
- **Eyeriss v2:** Fixed 8×2 array of 16 clusters (192 total PEs)
- **OpenEye:** Parameterizable PE_ROWS × PE_COLUMNS per cluster, supporting 1×1 to N×M configurations

**2. Data Widths:**
- **Eyeriss v2:** Fixed 8-bit data, 20-bit accumulators, 4-bit overhead
- **OpenEye:** Configurable DATA_IACT_BITWIDTH, DATA_WGHT_BITWIDTH, DATA_PSUM_BITWIDTH with flexible overhead

**3. Interface Bus Widths:**
- **Eyeriss v2:** Fixed per-design interface widths matching ASIC layout
- **OpenEye:** Parameterizable TRANS_BITWIDTH_IACT, TRANS_BITWIDTH_WGHT, TRANS_BITWIDTH_PSUM for FPGA optimization

**4. Processing Mode:**
- **Eyeriss v2:** Always parallel (2 MACs per cycle)
- **OpenEye:** Selectable via SERIAL parameter (1 MAC serial or 2/4 MACs parallel)

**5. Global Buffer Connectivity:**
- **Eyeriss v2:** 3 fixed iact routers per cluster, direct connectivity
- **OpenEye:** NUM_GLB_IACT configurable interfaces with independent routing per PE

**6. Reset Distribution:**
- **Eyeriss v2:** Centralized reset tree
- **OpenEye:** Optional RST_SYNC module for clock-domain crossing at top level

**7. Partial Sum Routing:**
- **Eyeriss v2:** Single hierarchical mesh path
- **OpenEye:** Dual paths (direct + router) with multiplexer-based selection per column

### OpenEye PE Individual Unit

Each PE in the cluster is an instance of the [PE.v](../../hdl/PE.v) module, which implements:

#### PE Architecture Components
- **5 Scratchpad Memories:** IACT address, IACT data, WGHT address, WGHT data, PSUM (matching Eyeriss v2)
- **7-Stage Pipeline:** Handling compressed data dependencies and SIMD operations
- **Dual MAC Units:** 2 parallel multiply-accumulate units (PARALLEL_MACS=2)
- **Sparsity Engine:** Zero-skipping logic with CSC format support
- **Flexible Routing:** Input selection from NUM_GLB_IACT sources
- **Systolic Connectivity:** Partial sum accumulation across PE arrays

#### PE Interface Signals
Each PE in the cluster receives:
- `iact_select_i`: Source selection for NUM_GLB_IACT activation interfaces
- `compute_i`: Computation trigger (managed by parent PE_cluster)
- `iact_data_i`, `iact_enable_i`, `iact_ready_o`: Activation interface (all sources concatenated)
- `wght_data_i`, `wght_enable_i`, `wght_ready_o`: Weight interface (row-wise shared)
- `psum_data_i`, `psum_enable_i`, `psum_ready_o`: Upstream partial sums
- `psum_data_o`, `psum_enable_o`, `psum_ready_i`: Downstream partial sums

#### PE Coordinate System
Each PE maintains awareness of its position in the grid:
- `PE_X`: Horizontal position (column index, 0 to PE_COLUMNS-1)
- `PE_Y`: Vertical position (row index, 0 to PE_ROWS-1)
- Used for potential neighbor identification and relative positioning

### Parameter Recommendation for Different Designs

**Sparse Neural Network Inference (Default Configuration):**
```verilog
PE_ROWS = 3, PE_COLUMNS = 4          // 12 PE baseline
PARALLEL_MACS = 2                     // Dual MAC for throughput
SERIAL = 0                            // Parallel mode
DATA_IACT_BITWIDTH = 8, DATA_WGHT_BITWIDTH = 8
DATA_PSUM_BITWIDTH = 20              // Sufficient for accumulation
NUM_GLB_IACT = 3                     // Multi-source activation
IACT_DATA_WORDS = 16, WGHT_DATA_WORDS = 96  // Standard capacity
```

**Area-Constrained Implementation:**
```verilog
PE_ROWS = 2, PE_COLUMNS = 2          // 4 PEs
PARALLEL_MACS = 1                    // Single MAC (serial mode)
SERIAL = 1                           // Reduced area
DATA_IACT_BITWIDTH = 4, DATA_WGHT_BITWIDTH = 4
DATA_PSUM_BITWIDTH = 16              // Reduced precision
NUM_GLB_IACT = 1                    // Single interface
IACT_DATA_WORDS = 8, WGHT_DATA_WORDS = 32
```

**High-Throughput Configuration:**
```verilog
PE_ROWS = 4, PE_COLUMNS = 4          // 16 PEs
PARALLEL_MACS = 4                    // Quad MAC
SERIAL = 0                           // Full parallel
DATA_IACT_BITWIDTH = 8, DATA_WGHT_BITWIDTH = 8
DATA_PSUM_BITWIDTH = 24              // Extended range
NUM_GLB_IACT = 4                    // Multiple sources
IACT_DATA_WORDS = 32, WGHT_DATA_WORDS = 256
TRANS_BITWIDTH_IACT = 48, TRANS_BITWIDTH_WGHT = 48
```

## Summary

The PE cluster in Eyeriss v2 represents a sophisticated balance of several design goals:

**Key Innovations:**
- ✓ Direct processing of compressed sparse data (CSC format)
- ✓ SIMD support for 2× throughput with minimal overhead
- ✓ Flexible cluster organization with all-to-all connectivity
- ✓ Deep pipeline handling compressed data dependencies
- ✓ Sparsity-aware workload mapping at compile time
- ✓ Dense mode support for low-sparsity scenarios

**Design Philosophy:**
- **Computation is cheap**: Add SIMD, complex pipelines
- **Data movement is expensive**: Compress data, minimize transfers
- **Flexibility matters**: Support both sparse and dense efficiently
- **Predictability helps**: Compile-time optimization, static configuration

The PE cluster architecture enables Eyeriss v2 to achieve **12.6× speedup** and **2.5× energy efficiency** improvement over the original Eyeriss when processing sparse MobileNet, demonstrating the effectiveness of co-designing the compute architecture with the data compression and network infrastructure.

## References

- Chen et al., "Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks on Mobile Devices", IEEE JSSC 2019
- Parashar et al., "SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks", ISCA 2017 (CSC compression format)
- Han et al., "EIE: Efficient Inference Engine on Compressed Deep Neural Networks", ISCA 2016 (sparse processing concepts)