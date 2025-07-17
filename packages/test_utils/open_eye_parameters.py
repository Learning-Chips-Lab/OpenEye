# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

import os
import test_utils.generic_test_utils as generic_test_utils


class OpenEyeParameters(object):
    """ Parameter class for the OpenEye accelerator.
    
    This class contains all the parameters for the OpenEye accelerator that
    are fixed before the accelerator is synthesized.
    
    """
    def __init__(self, serial = False):

        try:
            self.Clusters_Y = int(os.getenv("CLUSTER_ROWS"))
        except:
            self.Clusters_Y = 8
        try:
            self.NUM_GLB_IACT = int(os.getenv("NUM_GLB_IACT"))
        except:
            self.NUM_GLB_IACT = 3
        try:
            self.NUM_GLB_PSUM = int(os.getenv("NUM_GLB_PSUM"))
            self.PEs_X = int(os.getenv("NUM_GLB_PSUM"))
        except:
            self.NUM_GLB_PSUM = 4
            self.PEs_X = 4
        try:
            self.NUM_GLB_WGHT = int(os.getenv("NUM_GLB_WGHT"))
            self.PEs_Y = int(os.getenv("NUM_GLB_WGHT"))
        except:
            self.NUM_GLB_WGHT = 3
            self.PEs_Y = 3
        self.SERIAL = serial
        self.PARALLEL_MACS = 2

        self.IACT_Bitwidth = 8
        self.WGHT_Bitwidth = 8
        self.IACT_WOH_Bitwidth = self.IACT_Bitwidth + 4
        self.WGHT_WOH_Bitwidth = self.WGHT_Bitwidth + 4
        self.IACT_Addr_Bitwidth = 4
        self.WGHT_Addr_Bitwidth = 7
        self.PSUM_Bitwidth = 20
        self.IACT_Trans_Bitwidth = 24
        self.WGHT_Trans_Bitwidth = 24
        self.PSUM_Trans_Bitwidth = 20 * self.PARALLEL_MACS
        self.Clusters_X = 2
        self.PEs = self.PEs_X * self.PEs_Y
        self.Clusters = self.Clusters_X * self.Clusters_Y
        self.PE_Complete = self.PEs * self.Clusters
        self.Iacts_Addr_per_PE = 9
        self.Iacts_per_PE = 16
        self.Wghts_Addr_per_PE = 16
        self.Wghts_per_PE = 96 * 2
        self.Psums_per_PE  = 32
        self.Wght_Routers = self.PEs_Y
        self.Psum_Routers = self.PEs_X
        self.data_mode = 0
        self.autofunction = 0
        self.poolingmode = 1
        
        self.NUM_BUFFER = 32

        self.Iact_Mem_Addr_Words = 512
        self.Psum_Mem_Addr_Words = 384 * 2

        self.Router_Modes_IACT = 1
        self.Router_Modes_WGHT = 1
        self.Router_Modes_PSUM = 1

        self.poolingmode = 1
        
        self.DMA_Bit_AXI = 64
        self.FSM_CYCLE_BITWIDTH = 1024
        self.FSM_STATES = 9
        self.Iact_Router_Bits = 6
        self.Wght_Router_Bits = 1
        self.Psum_Router_Bits = 3


def get_oep(serial = False):
    openeye_parameter = OpenEyeParameters(serial)
    return openeye_parameter
