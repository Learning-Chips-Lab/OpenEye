# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""OpenEye hardware configuration parameters.

This module defines the OpenEyeParameters class which encapsulates all hardware
configuration parameters for the OpenEye neural network accelerator. These parameters
describe the fixed hardware architecture that is determined at synthesis time.

Key Features:
    - Configurable cluster and PE array dimensions
    - Hardware-specific bitwidth and memory size parameters
    - Support for both serial (DMA-based) and parallel communication modes
    - Environment variable overrides for key hardware parameters
    - Router and memory configuration for data paths

Hardware Architecture Overview:
    The OpenEye accelerator consists of a 2D grid of clusters, each containing
    a 2D array of Processing Elements (PEs). Data flows through three main paths:
    - Input Activations (IACT): Input feature maps
    - Weights (WGHT): Convolutional filters or dense layer weights
    - Partial Sums (PSUM): Intermediate and final computation results

Typical Usage:
    >>> params = OpenEyeParameters(serial=False)
    >>> print(f"Total PEs: {params.PE_Complete}")
    >>> print(f"Cluster grid: {params.Clusters_X}x{params.Clusters_Y}")

    >>> # Use environment variables to override defaults
    >>> os.environ["CLUSTER_ROWS"] = "4"
    >>> params = OpenEyeParameters()
"""

import os
import open_eye.generic_test_utils as generic_test_utils


class OpenEyeParameters(object):
    """Hardware configuration parameters for the OpenEye neural network accelerator.

    This class contains all hardware parameters that define the OpenEye accelerator
    architecture. These parameters are fixed at synthesis time and describe the
    physical structure of the accelerator including cluster layout, PE array dimensions,
    memory sizes, data bitwidths, and router configurations.

    The parameters can be customized via environment variables to support different
    hardware configurations without code changes. This is particularly useful for
    testing different accelerator sizes or configurations.

    Attributes:
        Clusters_X (int): Number of PE clusters in X (horizontal) dimension. Fixed at 2.
        Clusters_Y (int): Number of PE clusters in Y (vertical) dimension. Default 8,
                         overridable via CLUSTER_ROWS environment variable.
        PEs_X (int): Number of PEs per cluster in X dimension. Default 4,
                    overridable via NUM_GLB_PSUM environment variable.
        PEs_Y (int): Number of PEs per cluster in Y dimension. Default 3,
                    overridable via NUM_GLB_WGHT environment variable.
        PEs (int): Total PEs per cluster (PEs_X * PEs_Y).
        Clusters (int): Total number of clusters (Clusters_X * Clusters_Y).
        PE_Complete (int): Total PEs across entire accelerator (PEs * Clusters).

        NUM_GLB_IACT (int): Number of global input activation buffers/routers per cluster.
        NUM_GLB_WGHT (int): Number of global weight buffers/routers per cluster.
        NUM_GLB_PSUM (int): Number of global partial sum buffers/routers per cluster.

        SERIAL (bool): Communication mode flag. True for serial DMA-based communication,
                      False for parallel communication.
        PARALLEL_MACS (int): Number of parallel MAC operations per PE. Fixed at 2.

        IACT_Bitwidth (int): Bitwidth for input activation data values (8 bits, INT8).
        WGHT_Bitwidth (int): Bitwidth for weight data values (8 bits, INT8).
        IACT_WOH_Bitwidth (int): Bitwidth for input activations with overhead (12 bits).
        WGHT_WOH_Bitwidth (int): Bitwidth for weights with overhead (12 bits).
        IACT_Addr_Bitwidth (int): Bitwidth for input activation addresses (4 bits).
        WGHT_Addr_Bitwidth (int): Bitwidth for weight addresses (7 bits).
        PSUM_BITWIDTH (int): Bitwidth for partial sum values (20 bits).
        IACT_Trans_Bitwidth (int): Bitwidth for activation transmissions (24 bits).
        WGHT_Trans_Bitwidth (int): Bitwidth for weight transmissions (24 bits).
        PSUM_Trans_Bitwidth (int): Bitwidth for psum transmissions (40 bits for 2 MACs).

        Iacts_Addr_per_PE (int): Number of activation address storage locations per PE (9).
        Iacts_per_PE (int): Number of activation data storage locations per PE (16).
        Wghts_Addr_per_PE (int): Number of weight address storage locations per PE (16).
        Wghts_per_PE (int): Number of weight data storage locations per PE (192).
        Psums_per_PE (int): Number of partial sum storage locations per PE (32).

        Wght_Routers (int): Number of weight routers (equals PEs_Y).
        Psum_Routers (int): Number of partial sum routers (equals PEs_X).

        data_mode (int): Data transmission mode selector (0 for standard mode).
        autofunction (int): Auto-function configuration flag.
        poolingmode (int): Pooling operation mode selector (1 for enabled).

        NUM_BUFFER (int): Number of buffer entries for data staging (32).

        Iact_Mem_Addr_Words (int): Input activation memory size in words (512).
        Psum_Mem_Addr_Words (int): Partial sum memory size in words (768).

        Router_Modes_IACT (int): Number of routing modes for activation path.
        Router_Modes_WGHT (int): Number of routing modes for weight path.
        Router_Modes_PSUM (int): Number of routing modes for partial sum path.

        DMA_Bit_AXI (int): AXI bus width for DMA transfers (64 bits).
        FSM_CYCLE_BITWIDTH (int): Bitwidth for FSM cycle counter (1024 bits).
        FSM_STATES (int): Number of FSM states in control logic (9).
        Iact_Router_Bits (int): Bitwidth for activation router configuration (6 bits).
        Wght_Router_Bits (int): Bitwidth for weight router configuration (1 bit).
        Psum_Router_Bits (int): Bitwidth for partial sum router configuration (3 bits).

    """
    def __init__(self, serial = False):
        """Initialize OpenEye hardware configuration parameters.

        Creates a parameter object with default hardware configuration values.
        Key parameters can be overridden via environment variables to support
        different accelerator configurations for testing and deployment.

        Args:
            serial (bool): Communication mode flag. If True, configures for serial
                          DMA-based communication. If False (default), configures
                          for parallel communication. Default is False.

        Environment Variables:
            CLUSTER_ROWS: Overrides Clusters_Y (number of cluster rows)
            NUM_GLB_IACT: Overrides number of global activation buffers per cluster
            NUM_GLB_PSUM: Overrides PEs_X and NUM_GLB_PSUM (horizontal PE count)
            NUM_GLB_WGHT: Overrides PEs_Y and NUM_GLB_WGHT (vertical PE count)

        Note:
            The environment variable names reflect the correspondence between
            global buffers and PE dimensions:
            - NUM_GLB_PSUM determines PEs_X (partial sums flow horizontally)
            - NUM_GLB_WGHT determines PEs_Y (weights distributed vertically)

        """
        # === CLUSTER AND PE ARRAY DIMENSIONS ===
        # These parameters define the 2D grid structure of the accelerator

        # Number of cluster COLUMNS (X dimension)
        # Can be overridden by CLUSTER_COLUMNS environment variable
        try:
            self.Clusters_X = int(os.getenv("CLUSTER_COLUMNS"))
        except:
            self.Clusters_X = 2  # Default: 2 columns of clusters


        # Number of cluster rows (Y dimension)
        # Can be overridden by CLUSTER_ROWS environment variable
        try:
            self.Clusters_Y = int(os.getenv("CLUSTER_ROWS"))
        except:
            self.Clusters_Y = 8  # Default: 8 rows of clusters

        # Number of global input activation buffers per cluster
        # Can be overridden by NUM_GLB_IACT environment variable
        try:
            self.NUM_GLB_IACT = int(os.getenv("NUM_GLB_IACT"))
        except:
            self.NUM_GLB_IACT = 3  # Default: 3 global IACT buffers per cluster
        # Number of global partial sum buffers and horizontal PEs per cluster
        # NUM_GLB_PSUM corresponds to PEs_X since partial sums flow horizontally
        # Can be overridden by NUM_GLB_PSUM environment variable
        try:
            self.NUM_GLB_PSUM = int(os.getenv("NUM_GLB_PSUM"))
            self.PEs_X = int(os.getenv("NUM_GLB_PSUM"))  # Horizontal PE count
        except:
            self.NUM_GLB_PSUM = 4  # Default: 4 global PSUM buffers per cluster
            self.PEs_X = 4         # Default: 4 PEs horizontally per cluster

        # Number of global IACT RAM buffers
        try:
            self.IACT_RAM_CELLS = int(os.getenv("IACT_RAM_CELLS"))
        except:
            self.IACT_RAM_CELLS = 4  # Default: 4 global PSUM buffers per cluster

        # Number of global weight buffers and vertical PEs per cluster
        # NUM_GLB_WGHT corresponds to PEs_Y since weights are distributed vertically
        # Can be overridden by NUM_GLB_WGHT environment variable
        try:
            self.NUM_GLB_WGHT = int(os.getenv("NUM_GLB_WGHT"))
            self.PEs_Y = int(os.getenv("NUM_GLB_WGHT"))  # Vertical PE count
        except:
            self.NUM_GLB_WGHT = 3  # Default: 3 global WGHT buffers per cluster
            self.PEs_Y = 3         # Default: 3 PEs vertically per cluster

        # Number of Width of GLBs
        try:
            self.BUFFER_WIDTH = int(os.getenv("BUFFER_WIDTH"))
        except:
            self.BUFFER_WIDTH = 10

        # Number of possible branches in one single net
        try:
            self.BRANCHES = int(os.getenv("BRANCHES"))
        except:
            self.BRANCHES = 2  
        # Number of possible branches in one single net
        try:
            self.QUANT_AMOUNT = int(os.getenv("QUANT_AMOUNT"))
        except:
            self.QUANT_AMOUNT = 32  
        # Number of parallel rewrites to next layer
        try:
            self.TRANS_WORDS = int(os.getenv("TRANS_WORDS"))
        except:
            self.TRANS_WORDS = 8 
        # Partial sum bitwidth (wider to prevent overflow during accumulation)
        try:
            self.DATA_PSUM_BITWIDTH = int(os.getenv("DATA_PSUM_BITWIDTH"))
        except:
            self.DATA_PSUM_BITWIDTH = 20        # Partial sum: 20 bits (signed accumulator)

        # === COMMUNICATION MODE ===
        self.SERIAL = serial           # Serial (DMA) vs parallel communication mode

        # === DATAFLOW SELECTION ===
        # "row_stationary" (default): Eyeriss-style conv mapping, filter rows and
        # psums stationary in the PEs, iact routing via iact_choose patterns.
        # "output_stationary": GEMM mapping, PE row j is hard-wired to iact GLB
        # bank j (gemm_mode=1 in hardware) and each PE keeps its output tile
        # stationary in the local psum SPad. Overridable via DATAFLOW env var.
        self.DATAFLOW = os.getenv("DATAFLOW", "row_stationary")
        self.PARALLEL_MACS = 2         # Number of MAC units operating in parallel per PE

        # === DATA BITWIDTH CONFIGURATION ===
        # Define bitwidths for all data types in the accelerator

        # Base data bitwidths (INT8 for activations and weights)
        self.IACT_Bitwidth = 8         # Input activation data: 8 bits (INT8)
        self.WGHT_Bitwidth = 8         # Weight data: 8 bits (INT8)

        # Bitwidths with overhead (WOH = With OverHead)
        # Additional 4 bits for metadata (e.g., validity, address info)
        self.IACT_WOH_Bitwidth = self.IACT_Bitwidth + 4  # Activation + overhead: 12 bits
        self.WGHT_WOH_Bitwidth = self.WGHT_Bitwidth + 4  # Weight + overhead: 12 bits

        # Address bitwidths for memory indexing
        self.IACT_Addr_Bitwidth = 4    # Activation address: 4 bits (16 locations)
        self.WGHT_Addr_Bitwidth = 7    # Weight address: 7 bits (128 locations)

        # Transmission bitwidths for data movement across routers
        self.IACT_Trans_Bitwidth = 24  # Activation transmission: 24 bits
        self.WGHT_Trans_Bitwidth = 24  # Weight transmission: 24 bits
        self.PSUM_Trans_Bitwidth = self.DATA_PSUM_BITWIDTH * self.PARALLEL_MACS  # PSUM transmission: 40 bits (2 MACs)

        # === DERIVED DIMENSIONS ===
        # Calculate total cluster and PE counts from base parameters

        self.PEs = self.PEs_X * self.PEs_Y  # Total PEs per cluster (e.g., 4*3 = 12)
        self.Clusters = self.Clusters_X * self.Clusters_Y  # Total clusters (e.g., 2*8 = 16)
        self.PE_Complete = self.PEs * self.Clusters  # Total PEs in accelerator (e.g., 192)

        # === MEMORY SIZES PER PE ===
        # Define storage capacity for each PE's local memory

        self.Iacts_Addr_per_PE = 9     # Activation address entries per PE
        self.Iacts_per_PE = 16         # Activation data entries per PE
        self.Wghts_Addr_per_PE = 16    # Weight address entries per PE
        self.Wghts_per_PE = 96 * 2     # Weight data entries per PE (192 total)
        self.Psums_per_PE  = 32        # Partial sum entries per PE

        # === ROUTER CONFIGURATION ===
        # Number of routers matches PE dimensions for data distribution

        self.Wght_Routers = self.PEs_Y  # Weight routers per cluster (vertical distribution)
        self.Psum_Routers = self.PEs_X  # Partial sum routers per cluster (horizontal flow)

        # === OPERATIONAL MODES ===
        # Control flags for different operational configurations

        self.data_mode = 0             # Data transmission mode (0 = standard mode)
        self.autofunction = 0          # Auto-function enable flag
        self.poolingmode = 1           # Pooling operation mode (1 = enabled)

        # === BUFFER CONFIGURATION ===
        self.NUM_BUFFER = 32           # Number of buffer entries for data staging

        # === GLOBAL MEMORY SIZES ===
        # Memory sizes for cluster-level storage (in words)

        self.Iact_Mem_Addr_Words = 512      # Input activation memory: 512 words
        self.Psum_Mem_Addr_Words = 384 * 2  # Partial sum memory: 768 words

        # === ROUTER MODE CONFIGURATION ===
        # Number of different routing modes available for each data path

        self.Router_Modes_IACT = 1     # Activation router modes (1 = single mode)
        self.Router_Modes_WGHT = 1     # Weight router modes (1 = single mode)
        self.Router_Modes_PSUM = 1     # Partial sum router modes (1 = single mode)

        # === DMA AND CONTROL CONFIGURATION ===
        # Parameters for DMA transfers and FSM control logic

        self.DMA_Bit_AXI = 64          # AXI bus width for DMA transfers (64 bits)
        self.FSM_CYCLE_BITWIDTH = 1024 # Bitwidth for FSM cycle counter (1024 bits)
        self.FSM_STATES = 9            # Number of states in control FSM

        # === ROUTER BITWIDTH CONFIGURATION ===
        # Bitwidths for encoding router configurations

        self.Iact_Router_Bits = 6      # Activation router config: 6 bits (64 modes)
        self.Wght_Router_Bits = 1      # Weight router config: 1 bit (2 modes)
        self.Psum_Router_Bits = 3      # Partial sum router config: 3 bits (8 modes)

        self.IACT_RAM_CELLS_WORD_BITWIDTH = 64
        self.WGHT_RAM_CELLS_WORD_BITWIDTH = 24
        self.PSUM_RAM_CELLS_WORD_BITWIDTH = 64
        self.DATA_IACT_BITWIDTH = 8
        self.IACT_WORDS_IN_RAM = self.IACT_RAM_CELLS_WORD_BITWIDTH//self.DATA_IACT_BITWIDTH




def get_oep(serial = False):
    """Factory function to create an OpenEyeParameters instance.

    Convenience function that creates and returns an OpenEyeParameters object
    with the specified communication mode. This function provides a simpler
    interface for obtaining parameter objects in test and simulation code.

    Args:
        serial (bool): Communication mode flag. If True, creates parameters
                      configured for serial DMA-based communication. If False
                      (default), creates parameters for parallel communication.
                      Default is False.

    Returns:
        OpenEyeParameters: A fully initialized parameter object containing all
                          hardware configuration settings for the OpenEye accelerator.

    Example:
        >>> # Get parameters for parallel communication mode
        >>> params = get_oep(serial=False)
        >>> print(f"Accelerator has {params.PE_Complete} PEs")

        >>> # Get parameters for serial DMA mode
        >>> params_serial = get_oep(serial=True)
        >>> print(f"DMA bus width: {params_serial.DMA_Bit_AXI} bits")

    """
    openeye_parameter = OpenEyeParameters(serial)
    return openeye_parameter
