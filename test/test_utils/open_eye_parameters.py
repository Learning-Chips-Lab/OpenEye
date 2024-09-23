# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

import sys
import os
directory = (os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
sys.path.extend([directory, os.path.dirname(os.path.realpath(__file__))])
import test_utils.generic_test_utils as generic_test_utils


class OpenEyeParameters(object):
    """ Parameter class for the OpenEye accelerator.
    
    This class contains all the parameters for the OpenEye accelerator that
    are fixed before the accelerator is synthesized.
    
    """
    def __init__(self, serial = 0):
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
        self.PEs_X = 4
        self.PEs_Y = 3
        self.NUM_GLB_IACT = 3
        self.NUM_GLB_PSUM = 4
        self.NUM_GLB_WGHT = 3
        self.Clusters_X = 2
        self.Clusters_Y = 8
        self.PEs = self.PEs_X * self.PEs_Y
        self.Clusters = self.Clusters_X * self.Clusters_Y
        self.PE_Complete = self.PEs * self.Clusters
        self.Iacts_Addr_per_PE = 9
        self.Iacts_per_PE = 16
        self.Wghts_Addr_per_PE = 16
        self.Wghts_per_PE = 96 * 2
        self.Psums_per_PE  = 32
        self.Iact_Routers = 3
        self.Wght_Routers = self.PEs_Y
        self.Psum_Routers = self.PEs_X
        self.data_mode = 1
        self.autofunction = 0
        self.poolingmode = 1
        
        self.Iact_Mem_Addr_Words = 512
        self.Psum_Mem_Addr_Words = 384 * 2

        self.Router_Modes_IACT = 1
        self.Router_Modes_WGHT = 1
        self.Router_Modes_PSUM = 1

        self.poolingmode = 1

        self.DMA_Bits = 48
        self.FSM_CYCLE_BITWIDTH = 1024
        self.FSM_STATES = 9
        self.Iact_Router_Bits = 6
        self.Wght_Router_Bits = 1
        self.Psum_Router_Bits = 3

#Change to OpenEyeParameters

def make_vh_file(params):
    generic_test_utils.delete_files_in_directory('demo/')
    filename = 'demo/parameters.vh'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    txt_file = open(filename, 'w')
    txt_file.write("parameter PARALLEL_MACS  = "  + str(params.PARALLEL_MACS)+ ";\n")
    txt_file.write("\n")
    txt_file.write("parameter DATA_IACT_BITWIDTH  = "  + str(params.IACT_Bitwidth)+ ";\n")
    txt_file.write("parameter DATA_PSUM_BITWIDTH  = "  + str(params.PSUM_Bitwidth)+ ";\n")
    txt_file.write("parameter DATA_WGHT_BITWIDTH  = "  + str(params.WGHT_Bitwidth)+ ";\n")
    txt_file.write("\n")
    txt_file.write("parameter TRANS_BITWIDTH_IACT  = "  + str(params.IACT_Trans_Bitwidth)+ ";\n")
    txt_file.write("parameter TRANS_BITWIDTH_PSUM  = "  + str(params.PSUM_Trans_Bitwidth)+ ";\n")
    txt_file.write("parameter TRANS_BITWIDTH_WGHT = "  + str(params.WGHT_Trans_Bitwidth)+ ";\n")
    txt_file.write("\n")
    txt_file.write("parameter NUM_GLB_IACT  = "  + str(params.NUM_GLB_IACT)+ ";\n")
    txt_file.write("parameter NUM_GLB_PSUM  = "  + str(params.NUM_GLB_PSUM)+ ";\n")
    txt_file.write("parameter NUM_GLB_WGHT = "  + str(params.NUM_GLB_WGHT)+ ";\n")
    txt_file.write("\n")
    txt_file.write("parameter PE_ROWS  = "  + str(params.PEs_Y)+ ";\n")
    txt_file.write("parameter PE_COLUMNS  = "  + str(params.PEs_X)+ ";\n")
    txt_file.write("parameter PES = "  + str(params.PEs_Y * params.PEs_X)+ ";\n")
    txt_file.write("\n")
    txt_file.write("parameter CLUSTER_ROWS  = "  + str(params.Clusters_Y)+ ";\n")
    txt_file.write("parameter CLUSTER_COLUMNS  = "  + str(params.Clusters_X)+ ";\n")
    txt_file.write("parameter CLUSTERS = "  + str(params.Clusters_Y * params.Clusters_X)+ ";\n")
    txt_file.write("\n")
    txt_file.write("parameter IACT_PER_PE  = "  + str(params.Iacts_per_PE)+ ";\n")
    txt_file.write("parameter PSUM_PER_PE  = "  + str(params.Psums_per_PE)+ ";\n")
    txt_file.write("parameter WGHT_PER_PE = "  + str(params.Wghts_per_PE)+ ";\n")
    txt_file.write("\n")
    txt_file.write("parameter IACT_ADDR_PER_PE  = "  + str(params.Iacts_Addr_per_PE)+ ";\n")
    txt_file.write("parameter WGHT_ADDR_PER_PE  = "  + str(params.Wghts_Addr_per_PE)+ ";\n")
    txt_file.write("\n")
    txt_file.write("parameter IACT_MEM_ADDR_WORDS  = "  + str(params.Iact_Mem_Addr_Words)+ ";\n")
    txt_file.write("parameter IACT_MEM_ADDR_BITS  = $clog2(IACT_MEM_ADDR_WORDS);\n")
    txt_file.write("parameter IACT_FSM_CYCL_WORDS = IACT_PER_PE + IACT_ADDR_PER_PE;\n")
    txt_file.write("parameter WGHT_FSM_CYCL_WORDS = WGHT_PER_PE + WGHT_ADDR_PER_PE;\n")
    txt_file.write("parameter PSUM_MEM_ADDR_WORDS  = "  + str(params.Psum_Mem_Addr_Words)+ ";\n")
    txt_file.write("parameter PSUM_MEM_ADDR_BITS  = $clog2(PSUM_MEM_ADDR_WORDS);\n")
    txt_file.write("\n")
    txt_file.write("parameter ROUTER_MODES_IACT  = "  + str(params.Router_Modes_IACT)+ ";\n")
    txt_file.write("parameter ROUTER_MODES_WGHT  = "  + str(params.Router_Modes_WGHT)+ ";\n")
    txt_file.write("parameter ROUTER_MODES_PSUM = "  + str(params.Router_Modes_PSUM)+ ";\n")
    txt_file.write("\n")
    txt_file.write("parameter DMA_BITWIDTH  = "  + str(params.DMA_Bits)+ ";\n")
    txt_file.write("parameter FSM_CYCLE_BITWIDTH  = "  + str(params.FSM_CYCLE_BITWIDTH)+ ";\n")
    txt_file.write("parameter FSM_STATES = "  + str(params.FSM_STATES)+ ";\n")
    txt_file.close()

def create_vh_file(serial = 0):
    openeye_parameter = OpenEyeParameters(serial)
    make_vh_file(openeye_parameter)
    return openeye_parameter
