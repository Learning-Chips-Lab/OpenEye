# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import os
from .docstring_parser import parse_documentation

class TestDocstringParser(object):

    def parse_and_compare_adder_doc(self, text):
        doc = parse_documentation(text)
        print(f"Module Name: {doc.module_name}")
        print(f"Description: {doc.description}")
        print("Parameters:")
        for name, desc in doc.parameters.items():
            print(f"  {name}: {desc}")
        print("Ports:")
        for name, desc in doc.ports.items():
            print(f"  {name}: {desc}")

        assert doc.module_name == "adder"
        description = "A simple registered adder module.\n\nThe adder computes the sum of `summand_1_i` and `summand_2_i` when `adder_en_i` is high.\nThe result is stored in `sum_o` and is updated on the rising edge of `clk_i`.\nThe result is reset to 0 when `rst_ni` is low."
        assert doc.description == description

        assert doc.parameters["DATA_WIDTH_SUM"] == "Bitwidth of the sum output"

        assert doc.ports["clk_i"] == "Clock input"
        assert doc.ports["rst_ni"] == "Active low reset input"
        assert doc.ports["adder_en_i"] == "Adder enable input"
        assert doc.ports["summand_1_i"] == "First summand input"
        assert doc.ports["summand_2_i"] == "Second summand input"
        assert doc.ports["sum_o"] == "Sum output"


        
    def test_parse_tripleslash(self):
        text="""
/// Module: adder
///
/// A simple registered adder module.
///
/// The adder computes the sum of `summand_1_i` and `summand_2_i` when `adder_en_i` is high.
/// The result is stored in `sum_o` and is updated on the rising edge of `clk_i`.
/// The result is reset to 0 when `rst_ni` is low.
///
/// Parameters:
///   DATA_WIDTH_SUM: Bitwidth of the sum output
///
/// Ports:
///   clk_i: Clock input
///   rst_ni: Active low reset input
///   adder_en_i: Adder enable input
///   summand_1_i: First summand input
///   summand_2_i: Second summand input
///   sum_o: Sum output
///
// More Verilog code here
"""
        self.parse_and_compare_adder_doc(text)

    def test_parse_doubleslash_exclamation(self):
        text="""
timescale 1ns / 1ps
// Some Verilog code here
//! Module: adder
//!
//! A simple registered adder module.
//!
/// The adder computes the sum of `summand_1_i` and `summand_2_i` when `adder_en_i` is high.
/// The result is stored in `sum_o` and is updated on the rising edge of `clk_i`.
/// The result is reset to 0 when `rst_ni` is low.
//!
//! Parameters:
//!   DATA_WIDTH_SUM: Bitwidth of the sum output
//!
//! Ports:
//!   clk_i: Clock input
//!   rst_ni: Active low reset input
//!   adder_en_i: Adder enable input
//!   summand_1_i: First summand input
//!   summand_2_i: Second summand input
//!   sum_o: Sum output
//!
// More Verilog code here
"""
        self.parse_and_compare_adder_doc(text)

    def test_parse_adder_file(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        test_file = os.path.join(cur_dir, "..", "..", "hdl", "adder.v")
        with open(test_file, "r") as f:
            text = f.read()
            self.parse_and_compare_adder_doc(text)

    def parse_and_compare_module_name_from_doc(self, module_file):
        """This simple test parses the module name from the docstring of the
        given module file and compares it with the module file name.
        
        The rational is that we can be sure that the module docstring is parsed
        correctly, i.e., no error is thrown, and the module name is correctly
        extracted from the docstring.
        """
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        test_file = os.path.join(cur_dir, "..", "..", "hdl", module_file)
        with open(test_file, "r") as f:
            text = f.read()

        doc = parse_documentation(text)
        assert doc.module_name == module_file.split(".")[0]

    def test_parse_af_cluster_file(self):
        self.parse_and_compare_module_name_from_doc("af_cluster.v")

    def test_parse_bano_cluster_file(self):
        self.parse_and_compare_module_name_from_doc("bano_cluster.v")

    def test_parse_data_pipeline_file(self):
        self.parse_and_compare_module_name_from_doc("data_pipeline.v")

    def test_parse_delay_cluster_file(self):
        self.parse_and_compare_module_name_from_doc("delay_cluster.v")

    def test_parse_demux2_file(self):
        self.parse_and_compare_module_name_from_doc("demux2.v")

    def test_parse_multiplier_file(self):
        self.parse_and_compare_module_name_from_doc("multiplier.v")

    def test_parse_mux2_file(self):
        self.parse_and_compare_module_name_from_doc("mux2.v")

    def test_parse_mux_iact_file(self):
        self.parse_and_compare_module_name_from_doc("mux_iact.v")

    def test_parse_router_iact_file(self):
        self.parse_and_compare_module_name_from_doc("router_iact.v")

    def test_parse_router_psum_file(self):
        self.parse_and_compare_module_name_from_doc("router_psum.v")

    def test_parse_router_wght_file(self):
        self.parse_and_compare_module_name_from_doc("router_wght.v")
    
    def test_parse_varlenFIFO_file(self):
        self.parse_and_compare_module_name_from_doc("varlenFIFO.v")

    def test_parse_GLB_cluster_file(self):
        self.parse_and_compare_module_name_from_doc("GLB_cluster.v")

    def test_parse_OpenEye_Cluster_file(self):
        self.parse_and_compare_module_name_from_doc("OpenEye_Cluster.v")

    def test_parse_OpenEye_Parallel_file(self):
        self.parse_and_compare_module_name_from_doc("OpenEye_Parallel.v")

    def test_parse_OpenEye_Wrapper_file(self):
        self.parse_and_compare_module_name_from_doc("OpenEye_Wrapper.v")

if __name__ == "__main__":
    t = TestDocstringParser()
    t.test_parse_tripleslash()
    t.test_parse_doubleslash_exclamation()
    t.test_parse_adder_file()