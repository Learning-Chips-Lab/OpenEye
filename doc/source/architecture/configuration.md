(configuration)=
# Configuration of the NOC and Execution

## Overview

The OpenEye neural network accelerator requires precise configuration to orchestrate data flow across the Network-on-Chip (NoC) and coordinate execution across Processing Elements (PEs). This document describes how data and configuration are formatted, encoded, and transmitted to the hardware using the Python tools available in the project.

The configuration system is hierarchical:
1. **Register Configuration**: Hardware control parameters packed into DMA words
2. **Routing Configuration**: NoC router control signals for data flow
3. **Data Streams**: Input activations, weights, and biases packed into transport format
4. **Execution Control**: Skip flags and mode selection signals

---

## 1. Stream Dictionaries and Data Organization

### Stream Dictionary Architecture

The OpenEye system organizes all transmissions using three stream dictionaries defined in [src/open_eye/stream_dicts.py](../../src/open_eye/stream_dicts.py):

#### Serial Stream Dictionary (DMA-based transmission)
Used for real silicon and synthesis with single DMA channel:

```python
stream_serial_dict = {
  "status": 0,          # Working parameters and layer configuration
  "router_iact": 1,     # Input activation routing configuration
  "router_wght": 2,     # Weight routing configuration
  "router_psum": 3,     # Partial sum routing configuration
  "iact_data": 4,       # Input activation payload data
  "wght_data": 5,       # Weight payload data
  "psum_data": 6        # Partial sum and bias payload data
}
```

Each stream index corresponds to a transmission phase in the control flow sequence.

#### Parallel Stream Dictionary (Direct port access)
Used for simulation and direct hardware testing with simultaneous port access:

```python
stream_parallel_dict = {
  "status": 0,          # Unified status transmission
  "iact": 1,            # Input activation data (parallel port)
  "wght": 2,            # Weight data (parallel port)
  "psum": 3,            # Partial sum data (parallel port)
  "quantize": 4,        # Quantization parameters (scale, shift pairs)
  "offset": 5           # Output offset parameters
}
```

Parallel mode allows simultaneous transmission on multiple independent ports, useful for cocotb testbenches and validation.

#### Status Dictionary (Layer Configuration Parameters)

The "status" transmission contains 26 sub-fields encoded as a dictionary structure:

```python
status_dict = {
  "data_mode": 0,                    # Data operation mode (0=normal, 1=alternate)
  "realfactor": 1,                   # Fixed-point scale factor for computation
  "autofunction": 2,                 # Auto-function mode flag for layer fusion
  "poolingmode": 3,                  # Pooling operation type (0=none, 1=avg, 2=max)
  "needed_refreshes": 4,             # Total computation cycles required
  "used_X_cluster": 5,               # Number of X-dimension clusters active
  "used_Y_cluster": 6,               # Number of Y-dimension clusters active
  "needed_Iact_writes": 7,           # Number of input activation transmission cycles
  "used_psum_per_PE": 8,             # Partial sum accumulator width per PE
  "used_iact_addr_per_PE": 9,        # Iact address pointer count per PE
  "used_wght_addr_per_PE": 10,       # Weight address pointer count per PE
  "used_iact_per_PE": 11,            # Input activation count per PE
  "iact_addr_len": 12,               # Length of iact address transmission stream
  "iact_data_len": 13,               # Length of iact data transmission stream
  "strideX": 14,                     # Stride in X dimension (1-8)
  "strideY": 15,                     # Stride in Y dimension (1-8)
  "kernel_per_pe_cluster": 16,       # Number of kernels per PE cluster
  "skipIact": 17,                    # Skip flag: 1=reuse cached iact, 0=load new
  "skipWght": 18,                    # Skip flag: 1=reuse cached weights, 0=load new
  "skipPsum": 19,                    # Skip flag: 1=reuse cached psums, 0=load new
  "usePEs": 20,                      # PE enable bitmap (1 bit per PE)
  "router_iact": 21,                 # Iact router configuration word index
  "router_wght": 22,                 # Weight router configuration word index
  "router_psum": 23,                 # Psum router configuration word index
  "psum_delay": 24,                  # Partial sum computation delay cycles
  "needed_standing_cycles": 25       # Synchronization/pipeline stabilization cycles
}
```

---

## 2. Hardware Register Configuration Encoding

### Register Map and Packing

The hardware requires precise register configuration transmitted as a sequence of DMA words. This is implemented in [src/open_eye/regmap_pack.py](../../src/open_eye/regmap_pack.py).

#### Key Parameters

| Parameter | Bitwidth | AXI Word | Bit Position | Description |
|-----------|----------|----------|--------------|-------------|
| wght_cycles_reg | 8 | 0 | 0-7 | Weight transmission cycles needed |
| stride_x_reg | 3 | 0 | 8-10 | Horizontal stride (1-8) |
| stride_y_reg | 3 | 0 | 11-13 | Vertical stride (1-8) |
| skipIact_reg | 1 | 0 | 14 | Skip input activation loading |
| skipWght_reg | 1 | 0 | 15 | Skip weight loading |
| skipPsum_reg | 1 | 0 | 16 | Skip partial sum loading |
| psum_delay_reg | 4 | 0 | 17-20 | Pipeline delay for psums |
| kernel_per_pe_cluster_reg | 4 | 0 | 21-24 | Kernels per cluster |
| kernel_size | 4 | 0 | 25-28 | Kernel size (2-7) |
| x_lines_reg | 8 | 0 | 29-36 | Number of X lines |
| needed_wght_cycles_reg | 8 | 0 | 37-44 | Weight computation cycles |
| needed_cycles_reg | 18 | 0 | 45-62 | Total computation cycles |
| iact_converter_buffer_addr_max_cycles | 8 | 1 | 0-7 | Iact buffer addressing cycles |
| iact_channels_per_pe | 8 | 1 | 8-15 | Channels per PE |
| iact_size_y | 8 | 1 | 16-23 | Input height |
| iact_size_x | 8 | 1 | 24-31 | Input width |
| iact_needed_cycles | 11 | 1 | 32-42 | Iact computation cycles |
| choose_iact_buffer_output | 1 | 1 | 63 | Output buffer selection |
| needed_iact_buffer_words_reg | 13 | 3 | 0-12 | Iact buffer word count |
| add_up_reg | 2 | 3 | 13-14 | Accumulation mode |

#### Transmission Structure

Configuration is transmitted as 4 consecutive 64-bit AXI words:

```
Transmission 0: Control Parameters (stride, skip flags, cycles)
  Bits [0-62]:   Control registers
  Bits [63]:     Reserved

Transmission 1: Input Activation Configuration
  Bits [0-15]:   Buffer addressing parameters
  Bits [16-31]:  Input dimensions
  Bits [32-62]:  Timing and routing information
  Bit [63]:      Mode selection

Transmission 2: Layer Parameters and Output Control
  Bits [0]:      Output buffer selection
  Bits [1-46]:   Channel and storage parameters
  Bits [47-50]:  Iact/weight address lengths
  Bits [51]:     Output data valid signal
  Bits [52-63]:  Reserved

Transmission 3: Advanced Parameters
  Bits [0-12]:   Buffer word count
  Bits [13-14]:  Accumulation mode
  Bits [15-63]:  Reserved
```

#### Register Packing Implementation

```python
def pack_registers(values):
    """
    Pack register values into 4 DMA words according to the register map.

    Args:
        values: Dictionary of register names to values

    Returns:
        List of 4 64-bit integers ready for AXI transmission
    """
    words = [0] * 4  # Create 4 words

    for register in REGISTERS:  # REGISTERS is the register definition list
        reg_value = values[register['name']]
        bit_mask = (1 << register['width']) - 1

        # Mask value to register width and shift to position
        masked_value = reg_value & bit_mask
        shifted_value = masked_value << register['pos']

        # OR into correct transmission word
        words[register['trans']] |= shifted_value

    return words
```

Example:
```python
# Input register values
regs = {
    "wght_cycles_reg": 5,
    "stride_x_reg": 2,
    "stride_y_reg": 2,
    "skipIact_reg": 0,
    "skipWght_reg": 0,
    "skipPsum_reg": 1,
    # ... (40+ more registers)
}

# Generate DMA words
dma_words = pack_registers(regs)
# Returns: [0x12345678, 0x9ABCDEF0, 0x13579BDF, 0x24680ACE]

# These words are transmitted sequentially over AXI
```

---

## 3. Network-on-Chip (NoC) Router Configuration

### Router Architecture Overview

The NoC uses three types of routers to control data distribution:

```
Input Data (GLB)
    │
    ├──→ Iact Router (6-bit controls) ──┐
    │                                    ├──→ PE Cluster Array
    ├──→ Weight Router (1-bit controls) ─┤
    │                                    ├──→ Computation
    └──→ Psum Router (2-bit controls) ───┘
        │
        └──→ Output to GLB
```

Each router controls how data is distributed across the cluster array during different phases of computation.

### Input Activation (Iact) Router Configuration

**Router Bit Width**: 6 bits per router
**Values per 64-bit Word**: 10 routers (10 × 6 = 60 bits)

**Router Values and Meanings**:

| Value | Mode | Description |
|-------|------|-------------|
| 1 | Single Y-cluster | Single cluster in Y dimension (simple case) |
| 3 | Single PE per cluster (first) | Individual PE access mode, first cluster |
| 9 | Multi Y-cluster (first) | First cluster in Y-dimension group |
| 17 | Multi Y-cluster (last) | Last cluster in Y-dimension group |
| 25 | Multi Y-cluster (middle) | Middle cluster in Y-dimension group |
| 33 | Single PE per cluster (other) | Individual PE access mode, non-first cluster |

**Encoding Example**:

```python
def pack_iact_routers(router_values):
    """
    Pack input activation router values into DMA words.

    Args:
        router_values: List of up to 64 router configuration values

    Returns:
        List of DMA words with packed router configurations
    """
    dma_storage = []
    IACT_ROUTER_BITS = 6
    DMA_BITWIDTH = 64
    routers_per_word = DMA_BITWIDTH // IACT_ROUTER_BITS  # = 10

    for word_idx in range((len(router_values) + routers_per_word - 1) // routers_per_word):
        dma_word = 0
        for router_idx in range(routers_per_word):
            global_router_idx = word_idx * routers_per_word + router_idx
            if global_router_idx < len(router_values):
                router_val = router_values[global_router_idx]
                # Position this router's value in the word
                dma_word |= (router_val & 0x3F) << (IACT_ROUTER_BITS * router_idx)

        dma_storage.append(dma_word)

    return dma_storage

# Example: Configure 3 iact routers
router_config = [1, 9, 25]  # Single cluster, first, middle
dma_words = pack_iact_routers(router_config)
# Word 0: 0x_______19_0001 (bit layout: [router2:17, router1:9, router0:1])
```

### Weight (Wght) Router Configuration

**Router Bit Width**: 1 bit per router
**Values per 64-bit Word**: 64 routers

**Router Values and Meanings**:

| Value | Description |
|-------|-------------|
| 0 | Use local weight: Load from local memory (first cluster) |
| 1 | Forward weight: Pass weight data from previous cluster |

**Encoding Example**:

```python
def pack_wght_routers(router_values):
    """
    Pack weight router values into DMA words (1 bit each).

    Args:
        router_values: List of router values (each 0 or 1)

    Returns:
        List of DMA words
    """
    dma_storage = []
    WGHT_ROUTER_BITS = 1
    DMA_BITWIDTH = 64
    routers_per_word = DMA_BITWIDTH // WGHT_ROUTER_BITS  # = 64

    for word_idx in range((len(router_values) + routers_per_word - 1) // routers_per_word):
        dma_word = 0
        for router_idx in range(routers_per_word):
            global_router_idx = word_idx * routers_per_word + router_idx
            if global_router_idx < len(router_values):
                dma_word |= (router_values[global_router_idx] & 1) << router_idx

        dma_storage.append(dma_word)

    return dma_storage

# Example: Configure weight forwarding for 4 clusters
wght_routers = [0, 0, 1, 1]  # Load, Load, Forward, Forward
dma_word = pack_wght_routers(wght_routers)
# Result: 0x_____________0C = binary 1100 (router3:1, router2:1, router1:0, router0:0)
```

### Partial Sum (Psum) Router Configuration

**Router Bit Width**: 2 bits per router
**Values per 64-bit Word**: 32 routers

**Router Values and Meanings**:

| Value | Binary | Description |
|-------|--------|-------------|
| 0 | 00 | No routing: Cluster inactive |
| 2 | 10 | Pass-through: Middle cluster in Y-group, forward psum down |
| 3 | 11 | Final accumulation: Last cluster in Y-group |
| 4 | 100 | Output cluster: Single PE per cluster mode |
| 5 | 101 | First accumulation: First cluster in Y-group |

**Encoding Example**:

```python
def pack_psum_routers(router_values):
    """
    Pack partial sum router values into DMA words (2 bits each).

    Args:
        router_values: List of router values (0, 2, 3, 4, or 5)

    Returns:
        List of DMA words
    """
    dma_storage = []
    PSUM_ROUTER_BITS = 2
    DMA_BITWIDTH = 64
    routers_per_word = DMA_BITWIDTH // PSUM_ROUTER_BITS  # = 32

    for word_idx in range((len(router_values) + routers_per_word - 1) // routers_per_word):
        dma_word = 0
        for router_idx in range(routers_per_word):
            global_router_idx = word_idx * routers_per_word + router_idx
            if global_router_idx < len(router_values):
                dma_word |= (router_values[global_router_idx] & 0x3) << (PSUM_ROUTER_BITS * router_idx)

        dma_storage.append(dma_word)

    return dma_storage

# Example: Configure psum routing for vertical accumulation
psum_routers = [5, 2, 3]  # First cluster, middle, last
dma_word = pack_psum_routers(psum_routers)
# Bit layout: [router2(bits 4-5)=3, router1(bits 2-3)=2, router0(bits 0-1)=5]
# Result: 0x__________2D = binary 00101101 (3<<4 | 2<<2 | 5<<0)
```

---

## 4. Data Stream Structures and Bit-Packing

### Input Activation (Iact) Data Format

Input activations are feature maps (typically 3D arrays: [channels][height][width]) that need to be serialized and transmitted to the hardware.

#### Data Organization

```
Original: Feature map [C][H][W]
            ↓
Transpose: [W][H][C]  (for efficient serialization)
            ↓
Flatten:   [W*H*C]    (1D stream)
            ↓
Pack:      Group values by byte/word boundaries
            ↓
DMA:       Transmit as 64-bit words
```

#### Bit-Packing

```python
def pack_iact_data(feature_map, iact_bitwidth=8):
    """
    Pack input activation data into DMA words.

    Args:
        feature_map: 3D numpy array [channels][height][width]
        iact_bitwidth: Bits per activation value (typically 8 for INT8)

    Returns:
        List of DMA words ready for transmission
    """
    import numpy as np

    # Transpose to [width][height][channels] for efficient packing
    data = np.transpose(feature_map, axes=(2, 1, 0))
    flat_data = data.flatten()

    DMA_BITWIDTH = 64
    values_per_word = DMA_BITWIDTH // iact_bitwidth

    dma_storage = []
    for word_idx in range((len(flat_data) + values_per_word - 1) // values_per_word):
        dma_word = 0
        for val_idx in range(values_per_word):
            global_idx = word_idx * values_per_word + val_idx
            if global_idx < len(flat_data):
                # Convert to unsigned 8-bit if signed
                value = int(flat_data[global_idx])
                if value < 0:
                    value = (1 << iact_bitwidth) + value  # Two's complement

                # Pack into DMA word (LSB first)
                dma_word |= (value & ((1 << iact_bitwidth) - 1)) << (iact_bitwidth * val_idx)

        dma_storage.append(dma_word)

    return dma_storage

# Example: Pack 4×4 feature map with 3 channels
fmap = np.array([
    [[10, 20], [30, 40]],  # Channel 0
    [[11, 21], [31, 41]],  # Channel 1
    [[12, 22], [32, 42]]   # Channel 2
])  # Shape: (3, 2, 2)

dma_words = pack_iact_data(fmap)
# With 8-bit values and 64-bit words: 8 values per word
# Word 0: values[0:8] packed as [val0 | val1<<8 | val2<<16 | ... | val7<<56]
```

**Transport Bus Formats**:

| Format | Bitwidth | Values per Word | Use Case |
|--------|----------|-----------------|----------|
| INT8 | 8 | 8 | Standard quantized activations |
| INT4 | 4 | 16 | Ultra-low precision networks |
| INT16 | 16 | 4 | Higher precision intermediate results |
| FP32 | 32 | 2 | Full precision networks |

### Weight (Wght) Data Format

Weights are stored with both address pointers (indirect indexing) and data values for sparse processing.

#### Two-Component Structure

```
Weights = [Address SPAD] + [Data SPAD]
           │                │
           ├─ Pointers       └─ Actual weight values
           │  to weight          (sparse format)
           │  data locations
           └─ 7-16 bits wide    └─ 8-24 bits wide
```

#### Data Organization by PE

```python
def pack_wght_data(weights, wght_bitwidth=8, sparse=False):
    """
    Pack weight data for transmission to PEs.

    Args:
        weights: Dict or multi-dimensional array of weight matrices
        wght_bitwidth: Bits per weight value (typically 8 for INT8)
        sparse: If True, store non-zero values with position metadata

    Returns:
        Dict with 'addr' (address SPAD) and 'data' (data SPAD) lists
    """

    wght_storage = {"addr": [], "data": []}
    DMA_BITWIDTH = 64
    values_per_word = DMA_BITWIDTH // wght_bitwidth

    # For sparse encoding:
    # Each weight word contains [count_bits | weight_value]
    # count_bits = number of zero columns skipped before this weight

    for pe_idx, pe_weights in enumerate(weights):
        # Convert to flat array and encode sparsity if needed
        flat_weights = pe_weights.flatten()

        if sparse:
            # Find non-zero positions and encode gaps
            nonzero_indices = np.where(flat_weights != 0)[0]
            weight_stream = []
            last_idx = 0

            for nz_idx in nonzero_indices:
                gap = nz_idx - last_idx
                weight_value = int(flat_weights[nz_idx])
                # Pack: [gap | weight_value]
                packed = (gap << wght_bitwidth) | (weight_value & ((1 << wght_bitwidth) - 1))
                weight_stream.append(packed)
                last_idx = nz_idx + 1
        else:
            weight_stream = flat_weights

        # Pack address pointers (where weights start in data SPAD)
        addr_word = 0
        current_addr = 0
        for addr_idx, weight in enumerate(weight_stream):
            if addr_idx % (DMA_BITWIDTH // 7) == 0:  # Address is 7 bits
                wght_storage["addr"].append(addr_word)
                addr_word = 0
                current_addr = 0

            addr_word |= (current_addr & 0x7F) << (7 * (addr_idx % (DMA_BITWIDTH // 7)))
            current_addr += 1

        # Pack weight data
        data_word = 0
        for val_idx, weight in enumerate(weight_stream):
            if val_idx % values_per_word == 0 and val_idx > 0:
                wght_storage["data"].append(data_word)
                data_word = 0

            weight_int = int(weight) & ((1 << wght_bitwidth) - 1)
            data_word |= weight_int << (wght_bitwidth * (val_idx % values_per_word))

        if weight_stream:  # Flush remaining data
            wght_storage["data"].append(data_word)

    return wght_storage

# Example: Pack 3×3 convolution weights (9 weights, 8-bit each)
weights = np.array([
    [-1, 2, -3],
    [4, -5, 6],
    [-7, 8, 9]
])

wght_data = pack_wght_data(weights)
# Returns:
# {'addr': [0x1234567], 'data': [0x0908070605040302, 0x00000000000000FF]}
```

### Partial Sum (Psum) and Bias Data Format

Partial sums (output accumulators) are initialized with bias values and updated during computation.

#### Bias Encoding

```python
def pack_psum_bias(biases, PSUM_BITWIDTH=20):
    """
    Pack bias values for partial sum initialization.

    Args:
        biases: 1D array of bias values (one per output filter)
        PSUM_BITWIDTH: Bits per accumulator (typically 20)

    Returns:
        List of DMA words with packed bias values
    """
    import numpy as np

    psum_storage = []
    DMA_BITWIDTH = 64
    values_per_word = DMA_BITWIDTH // PSUM_BITWIDTH

    for bias_idx in range(len(biases)):
        # Convert to two's complement if needed
        bias_val = int(biases[bias_idx])
        if bias_val < 0:
            bias_val = (1 << PSUM_BITWIDTH) + bias_val

        word_idx = bias_idx // values_per_word
        position_in_word = bias_idx % values_per_word

        # Extend storage if needed
        while len(psum_storage) <= word_idx:
            psum_storage.append(0)

        # Pack into word
        mask = (1 << PSUM_BITWIDTH) - 1
        psum_storage[word_idx] |= (bias_val & mask) << (PSUM_BITWIDTH * position_in_word)

    return psum_storage

# Example: Pack 4 bias values (20-bit each)
biases = np.array([100, -50, 200, -150])
psum_words = pack_psum_bias(biases)
# Word 0: bias[0] at bits [0-19], bias[1] at bits [20-39], bias[2] at bits [40-59]
# Word 1: bias[3] at bits [0-19]
```

---

## 5. Layer Mapper Orchestration

### Layer Mapping Workflow

The ConvMapper class ([src/open_eye/conv_mapper.py](../../src/open_eye/conv_mapper.py)) coordinates the complete configuration process:

```
Input Layer Parameters
    ↓
Determine PE Allocation (which PEs compute)
    ↓
Calculate Skip Flags (reuse cached data?)
    ↓
Pack Register Configuration (44 parameters)
    ↓
Generate PE Bitmap (1 bit per PE, indicates active)
    ↓
Configure Iact Router (how to distribute input)
    ↓
Configure Wght Router (weight forwarding pattern)
    ↓
Configure Psum Router (partial sum accumulation)
    ↓
Append Data Streams (iact, wght, psum payloads)
    ↓
Hardware Execution
```

### Complete Configuration Example

```python
def write_layer_config(params, layer_params, layer_repetition):
    """
    Generate complete configuration for one layer execution.

    Args:
        params: Hardware parameters (bit widths, dimensions, etc.)
        layer_params: Layer-specific parameters (stride, kernel size, etc.)
        layer_repetition: Iteration number (for skip flag logic)

    Returns:
        List of DMA words ready for transmission
    """
    from regmap_pack import pack_registers
    from router_pack import pack_iact_routers, pack_wght_routers, pack_psum_routers
    from stream_pack import pack_iact_data, pack_wght_data, pack_psum_bias

    dma_storage = []

    # Step 1: Determine skip flags based on data reuse patterns
    iact_transmissions_needed = calculate_iact_transmissions(layer_params)

    if (layer_repetition % iact_transmissions_needed) == 0:
        skipIact = 0  # Load new input activations
    else:
        skipIact = 1  # Reuse cached iact

    # Similar logic for weights and psums
    skipWght = 0 if (layer_repetition % wght_transmissions_needed) == 0 else 1
    skipPsum = 0 if (layer_repetition % psum_transmissions_needed) == 0 else 1

    # Step 2: Generate PE allocation bitmap
    computing_pes = 0
    pe_count = 0
    for cluster_x in range(params.clusters_X):
        for cluster_y in range(params.clusters_Y):
            for pe_y in range(params.PEs_Y):
                for pe_x in range(params.PEs_X):
                    if should_compute(cluster_x, cluster_y, pe_y, pe_x, layer_params):
                        computing_pes |= (1 << pe_count)
                    pe_count += 1

    # Step 3: Pack all 44 registers into 4 DMA words
    register_values = {
        "wght_cycles_reg": layer_params.wght_transmission_cycles,
        "stride_x_reg": layer_params.stride[1],
        "stride_y_reg": layer_params.stride[0],
        "skipIact_reg": skipIact,
        "skipWght_reg": skipWght,
        "skipPsum_reg": skipPsum,
        "psum_delay_reg": calculate_pipeline_delay(layer_params),
        "kernel_per_pe_cluster_reg": layer_params.kernels_per_cluster,
        # ... (40 more registers)
    }

    dma_words = pack_registers(register_values)
    dma_storage.extend(dma_words)

    # Step 4: Append PE bitmap (split across multiple words if needed)
    total_pes = params.clusters_X * params.clusters_Y * params.PEs_X * params.PEs_Y
    bitstring = format(computing_pes, f"0{total_pes}b")[::-1]  # Reverse for LSB-first

    for segment_idx in range((total_pes + 63) // 64):
        start = segment_idx * 64
        end = min(start + 64, total_pes)
        segment = bitstring[start:end]
        dma_storage.append(int(segment[::-1], 2))  # Reverse back for transmission

    # Step 5: Append router configurations
    if not skipIact:  # Only configure routers if loading new data
        iact_routers = calculate_iact_routing(layer_params)
        dma_storage.extend(pack_iact_routers(iact_routers))

        wght_routers = calculate_wght_routing(layer_params)
        dma_storage.extend(pack_wght_routers(wght_routers))

        psum_routers = calculate_psum_routing(layer_params)
        dma_storage.extend(pack_psum_routers(psum_routers))

    # Step 6: Append data streams
    if not skipIact:
        iact_data = prepare_iact_data(layer_params)
        dma_storage.extend(pack_iact_data(iact_data))

    if not skipWght:
        wght_addr, wght_data = prepare_wght_data(layer_params)
        dma_storage.extend(pack_wght_data(wght_addr, wght_data))

    if not skipPsum:
        psum_bias = prepare_psum_bias(layer_params)
        dma_storage.extend(pack_psum_bias(psum_bias))

    return dma_storage

# Usage
config_words = write_layer_config(hardware_params, conv_layer_params, repetition=0)
# Now send config_words over DMA to hardware
```

---

## 6. Quantization and Offset Parameters

### Quantization Encoding

For post-computation quantization (convert 20-bit psums to 8-bit output):

```python
def pack_quantize_params(scale, shift, num_filters=32):
    """
    Pack quantization scale and shift parameters.

    Quantization formula: output = (psum * scale) >> shift

    Args:
        scale: List of scale factors (typically 16-26 bits each)
        shift: List of shift amounts (typically 0-31)
        num_filters: Number of output filters

    Returns:
        List of DMA words (16 words for 32 filters, 2 per word)
    """
    dma_storage = []
    DMA_BITWIDTH = 64

    for filter_pair_idx in range((num_filters + 1) // 2):
        dma_word = 0

        # Pack two [scale, shift] pairs per word
        for pair_offset in range(2):
            filter_idx = filter_pair_idx * 2 + pair_offset
            if filter_idx < num_filters:
                # First scale/shift pair: bits [0-24] / [25-30]
                scale_shift = filter_pair_idx * 32
                shift_shift = scale_shift + 25

                if pair_offset == 1:
                    scale_shift = 32
                    shift_shift = 57

                scale_val = int(scale[filter_idx]) & 0x1FFFFFF
                shift_val = int(shift[filter_idx]) & 0x1F

                dma_word |= scale_val << scale_shift
                dma_word |= shift_val << shift_shift

        dma_storage.append(dma_word)

    return dma_storage

# Example: Quantize 4 filters
scales = [1000, 1050, 980, 1020]
shifts = [10, 10, 10, 10]
quantize_words = pack_quantize_params(scales, shifts, num_filters=4)
# Word 0: scale[0] at bits [0-24], shift[0] at bits [25-30],
#         scale[1] at bits [32-56], shift[1] at bits [57-62]
```

### Offset/Bias Encoding

```python
def pack_offset_params(offsets, num_filters=32):
    """
    Pack output offset parameters (for per-channel shifts).

    Args:
        offsets: List of 8-bit offset values
        num_filters: Number of output channels

    Returns:
        List of DMA words (4 words for 32 channels, 8 per word)
    """
    dma_storage = []
    DMA_BITWIDTH = 64
    offsets_per_word = DMA_BITWIDTH // 8  # 8 offsets per 64-bit word

    for word_idx in range((num_filters + offsets_per_word - 1) // offsets_per_word):
        dma_word = 0
        for offset_idx in range(offsets_per_word):
            global_idx = word_idx * offsets_per_word + offset_idx
            if global_idx < num_filters:
                offset_val = int(offsets[global_idx]) & 0xFF
                dma_word |= offset_val << (8 * offset_idx)

        dma_storage.append(dma_word)

    return dma_storage
```

---

## 7. Execution Control Signals

### PE Enable Bitmap

The system uses a bitmap to enable/disable specific PEs:

```python
def create_pe_bitmap(active_pes, total_pes):
    """
    Create bitmap indicating which PEs should execute.

    Args:
        active_pes: List of (cluster_x, cluster_y, pe_y, pe_x) tuples
        total_pes: Total PE count (e.g., 8 clusters × 12 PEs = 96)

    Returns:
        Integer bitmap with 1 bit per PE
    """
    bitmap = 0
    for idx, (cl_x, cl_y, pe_y, pe_x) in enumerate(active_pes):
        pe_number = cl_x * CLUSTERS_Y * PES_PER_CLUSTER_Y * PES_PER_CLUSTER_X + \
                    cl_y * PES_PER_CLUSTER_Y * PES_PER_CLUSTER_X + \
                    pe_y * PES_PER_CLUSTER_X + pe_x
        bitmap |= (1 << pe_number)

    return bitmap

# Example: Enable PEs 0, 2, 4, 6 in a 12-PE system
active_pes = [(0, 0, 0, 0), (0, 0, 0, 2), (0, 0, 1, 0), (0, 0, 1, 2)]
bitmap = create_pe_bitmap(active_pes, total_pes=12)
# Result: 0x555 = binary 010101010101 (alternate PEs enabled)
```

### Skip Flags

Three skip flags control data reuse across multiple layer iterations:

```python
skip_flags = {
    "skipIact": (layer_repetition % iact_iterations) != 0,  # True = reuse, False = load
    "skipWght": (layer_repetition % wght_iterations) != 0,  # True = reuse, False = load
    "skipPsum": (layer_repetition % psum_iterations) != 0,  # True = reuse, False = load
}
```

**Skip Flag Usage Pattern**:
- Layer has 2 different iact sizes, 1 wght set, 3 psum sets
- Iteration 0: skipIact=0, skipWght=0, skipPsum=0 → Load all
- Iteration 1: skipIact=1, skipWght=1, skipPsum=1 → Reuse all
- Iteration 2: skipIact=0, skipWght=1, skipPsum=1 → Load new iact only
- Iteration 3: skipIact=1, skipWght=1, skipPsum=0 → Load new psum only

---

## 8. Communication Modes: Serial vs. Parallel

### Serial Mode (Production)

Used for real silicon and synthesis:

```
Clock  │___|‾‾‾|___|‾‾‾|___|‾‾‾|___|‾‾‾|___|‾‾‾|
DMA    ╞═══════════════════════════════════════════╡
Valid  │   ─────────────────────────────────────
       │   Config    Router  Iact   Wght   Psum
       │   Cycle 0-3 Cycle4-6 Cycle7...
```

Single AXI DMA channel carries all traffic sequentially:
1. Register configuration (4 words)
2. PE bitmap (variable words)
3. Router configurations (Iact: ~7 words, Wght: 1 word, Psum: 2 words)
4. Data streams (Iact: variable, Wght: variable, Psum: variable)

### Parallel Mode (Simulation)

Used for cocotb testbenches and direct validation:

```
Clock  │___|‾‾‾|___|‾‾‾|___|‾‾‾|___|‾‾‾|___|‾‾‾|
       │
Iact   ╞═══════════════════════════════════════════╡
Valid  │   ─────────────────────────────────────
       │
Wght   ╞═══════════════════════════════════════════╡
Valid  │   ─────────────────────────────────────
       │
Psum   ╞═══════════════════════════════════════════╡
Valid  │   ─────────────────────────────────────
       │
       Simultaneous transmission on multiple ports
```

Parallel ports allow simultaneous transmission:
- Input activation port carries iact data
- Weight port carries wght data
- Partial sum port carries psum/bias data

---

## 9. Data Format Summary Table

| Data Type | Bitwidth | Values/64-bit | Format | Example |
|-----------|----------|---------------|--------|---------|
| Input Activation (Iact) | 8 | 8 | INT8 unsigned or signed | -128 to 127 |
| Weight | 8 | 8 | INT8 signed | -128 to 127 |
| Partial Sum (Psum) | 20 | 3 | INT20 two's complement | -524288 to 524287 |
| Scale Factor | 25 | 2.56 | Unsigned fixed-point | 0 to 33554431 |
| Shift Amount | 5 | 12.8 | Unsigned shift count | 0 to 31 |
| Bias | 20 | 3 | INT20 two's complement | -524288 to 524287 |
| Router Value (Iact) | 6 | 10.67 | Enumerated mode | 1, 3, 9, 17, 25, 33 |
| Router Value (Wght) | 1 | 64 | Binary mode | 0, 1 |
| Router Value (Psum) | 2 | 32 | Enumerated state | 0, 2, 3, 4, 5 |

---

## 10. Python Tool Usage Examples

### Using ConvMapper for Complete Configuration

```python
from src.open_eye.conv_mapper import ConvMapper
from src.open_eye.pe_cluster_test_utils import OpenEyeParameters
import numpy as np

# Initialize hardware parameters
hw_params = OpenEyeParameters(
    Clusters_X=2,
    Clusters_Y=2,
    PEs_X=2,
    PEs_Y=3,
    IACT_Bitwidth=8,
    WGHT_Bitwidth=8,
    PSUM_BITWIDTH=20
)

# Initialize layer parameters
layer_params = {
    'input_channels': 32,
    'output_channels': 64,
    'kernel_size': 3,
    'stride': (1, 1),
    'padding': (1, 1),
    'feature_map_size': (56, 56),
    'input_data': np.random.randint(-128, 127, (32, 56, 56), dtype=np.int8),
    'weights': np.random.randint(-128, 127, (64, 32, 3, 3), dtype=np.int8),
    'biases': np.random.randint(-1000, 1000, (64,), dtype=np.int32),
}

# Create mapper and generate configuration
mapper = ConvMapper(hw_params)
config_dma_words = mapper.write_working_parameters(hw_params, layer_params, layer_repetition=0)

# Append data streams
config_dma_words.extend(mapper.write_iact(hw_params, layer_params, 0))
config_dma_words.extend(mapper.write_wght(hw_params, layer_params, 0))
config_dma_words.extend(mapper.write_psum(hw_params, layer_params, 0))

# Now send config_dma_words over AXI to hardware
for idx, word in enumerate(config_dma_words):
    print(f"DMA Word {idx}: 0x{word:016X}")
```

### Direct Data Packing

```python
from src.open_eye.regmap_pack import pack_registers
from src.open_eye.stream_pack import pack_iact_data, pack_wght_data, pack_psum_bias
import numpy as np

# Pack register configuration
register_dict = {
    'stride_x_reg': 1,
    'stride_y_reg': 1,
    'wght_cycles_reg': 5,
    'needed_cycles_reg': 100,
    # ... (40 more registers)
}
config_words = pack_registers(register_dict)

# Pack iact data
iact_feature_map = np.random.randint(-128, 127, (32, 56, 56), dtype=np.int8)
iact_words = pack_iact_data(iact_feature_map, iact_bitwidth=8)

# Pack weight data
weights = np.random.randint(-128, 127, (64, 32, 3, 3), dtype=np.int8)
wght_dict = pack_wght_data(weights)

# Pack bias/psum data
biases = np.random.randint(-1000, 1000, (64,), dtype=np.int32)
psum_words = pack_psum_bias(biases, PSUM_BITWIDTH=20)

# Complete DMA transmission
complete_config = config_words + iact_words + wght_dict['data'] + psum_words
```

---

## 11. Troubleshooting Configuration

### Common Configuration Errors

**Stride Out of Range**
```
Error: stride_x_reg value 15 exceeds 3-bit maximum (0-7)
Fix: Ensure stride values are in [1, 2, 4, 8] for standard networks
```

**PE Bitmap Mismatch**
```
Error: PE bitmap has 96 bits but hardware has 24 PEs
Fix: Calculate PE count as clusters_X × clusters_Y × PEs_per_cluster
     Verify bitmap generation matches PE ordering
```

**Router Configuration Incomplete**
```
Error: Provided 8 iact routers but layer needs 12
Fix: Calculate required routers from cluster dimensions
     One router per cluster in typical mapping
```

**Data Format Mismatch**
```
Error: Input activation values exceed 8-bit range
Fix: Ensure activations are quantized to [-128, 127] before packing
     Use two's complement encoding for signed values
```

---

## References

- [src/open_eye/stream_dicts.py](../../src/open_eye/stream_dicts.py) - Stream dictionary definitions
- [src/open_eye/regmap_pack.py](../../src/open_eye/regmap_pack.py) - Register packing functions
- [src/open_eye/conv_mapper.py](../../src/open_eye/conv_mapper.py) - Complete layer mapping
- [src/open_eye/layer_execution_state.py](../../src/open_eye/layer_execution_state.py) - Execution tracking
- [test/cocotb_PE_cluster/test_PE_CLUSTER.py](../../test/cocotb_PE_cluster/test_PE_CLUSTER.py) - Hardware testbench examples
