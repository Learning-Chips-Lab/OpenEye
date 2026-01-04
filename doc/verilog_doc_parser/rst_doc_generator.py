# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import os
import warnings
from subprocess import Popen, PIPE

import logging
logger = logging.getLogger("doc")

from textwrap import indent

from docstring_parser import parse_documentation
from moduledef_parser import parse_module_def


def generate_rst(doc, module_def):
    """
    Generate RST documentation for a Verilog module.

    Safely handles missing parameters/ports that may be referenced in docstrings
    but not actually defined in the module.

    Args:
        doc: ModuleDocumentation object from parse_documentation()
        module_def: ModuleDefinition object from parse_module_def()

    Returns:
        String containing the RST documentation
    """
    docstring = ""

    title = f"{doc.module_name}"
    docstring += title + "\n"
    docstring += "=" * len(title) + "\n\n"

    docstring += f".. verilog:module:: module {doc.module_name}("
    for port, desc in doc.ports.items():
        docstring += f"{port}, "
    docstring = docstring[:-2] + ")\n\n"

    indented_description = indent(doc.description, "    ")

    docstring += f"{indented_description}\n\n"

    image_name = f"__{doc.module_name}.png"
    docstring += f"    .. image:: {image_name} \n\n"

    # Add parameters - skip any that don't exist in module_def
    for param, desc in doc.parameters.items():
        if param not in module_def.parameters:
            logger.debug(f"Parameter {param} documented but not found in module definition for {doc.module_name}")
            continue

        param_type = module_def.parameters[param].ptype
        param_default = module_def.parameters[param].default_value
        docstring += f"    .. verilog:parameter:: parameter {param_type} {param} = {param_default}\n\n"
        docstring += f"        {desc}\n\n"

    # Add ports - skip any that don't exist in module_def
    for port, desc in doc.ports.items():
        if port not in module_def.ports:
            logger.debug(f"Port {port} documented but not found in module definition for {doc.module_name}")
            continue

        port_direction = module_def.ports[port].direction
        port_sign = module_def.ports[port].sign
        port_type = module_def.ports[port].type
        port_upper = module_def.ports[port].range_upper
        port_lower = module_def.ports[port].range_lower

        docstring += f"    .. verilog:port:: {port_direction} {port_sign} {port_type} {port} "
        if port_upper!= 1 or port_lower != 0:
            docstring += f"[{port_upper}:{port_lower}]\n\n"
        else:
            docstring += "\n\n"
        docstring += f"        {desc}\n\n"

    return docstring


def generate_rst_file(input_filename):
    with open(input_filename, "r") as f_input:
        text = f_input.read()
        doc = parse_documentation(text)
        module_def = parse_module_def(text)
        rst = generate_rst(doc, module_def)
        doc_dir = __file__.split("/")[:-2]
        doc_dir.extend(["source", "verilog"])
        output_filename = "/".join(doc_dir) + f"/{doc.module_name}.rst"
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, "w") as f_output:
            f_output.write(rst) 

def generate_svg_file(input_filename):
    doc_dir = __file__.split("/")[:-2]
    doc_dir.extend(["source", "verilog"])

    cmd = [
        'symbolator',
        '-i', input_filename,
        '-t',
        '-o', "/".join(doc_dir) + f"/",
        '-f', 'PNG'
    ]
    p = Popen(cmd, stdout=PIPE, stdin=PIPE, stderr=PIPE)
    p.communicate()

def generate_index_rst(files):
    """
    Generate/update the Verilog modules index.rst file with all generated modules.

    This function creates a Sphinx toctree that includes all Verilog module documentation
    files, allowing them to be properly indexed in the documentation.

    Args:
        files: List of Verilog filenames (e.g., ['adder.v', 'multiplier.v'])
    """
    docstring = ""
    docstring += "Verilog Modules\n"
    docstring += "===============\n\n"
    docstring += ".. toctree::\n"
    docstring += "    :maxdepth: 2\n"
    docstring += "    :caption: Modules:\n\n"

    # Extract module names from filenames and sort them
    module_names = sorted([os.path.splitext(file)[0] for file in files])

    for module_name in module_names:
        docstring += f"    {module_name}\n"

    doc_dir = __file__.split("/")[:-2]
    doc_dir.extend(["source", "verilog"])
    output_filename = "/".join(doc_dir) + "/index.rst"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with open(output_filename, "w") as f_output:
        f_output.write(docstring)

    logger.info(f"Generated Verilog modules index with {len(module_names)} modules at {output_filename}")

if __name__ == "__main__":
    hdl_dir = __file__.split("/")[:-3]
    hdl_dir.extend(["hdl"])
    hdl_dir = "/".join(hdl_dir)
    generated_doc_files = []

    for root, dirs, files in os.walk(hdl_dir):
        for filename in sorted(files):
            if filename.endswith(".v"):
                # Try to generate RST (this is critical - file must be generated)
                try:
                    generate_rst_file(os.path.join(root, filename))
                    generated_doc_files.append(filename)
                except Exception as e:
                    warnings.warn(f"Error processing {filename}: {e}")
                    logger.debug(f"{e}")
                    continue

                # Try to generate SVG diagram (this is optional - don't fail if it doesn't work)
                try:
                    generate_svg_file(os.path.join(root, filename))
                except Exception as e:
                    # SVG generation is optional, just log it
                    logger.debug(f"Could not generate SVG for {filename}: {e}")
                    # Don't warn or continue - we still want to add to index

    # Always generate index with the files that were successfully processed
    generate_index_rst(generated_doc_files)