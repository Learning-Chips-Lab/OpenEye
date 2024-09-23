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

    for param, desc in doc.parameters.items():
        param_type = module_def.parameters[param].ptype
        param_default = module_def.parameters[param].default_value
        docstring += f"    .. verilog:parameter:: parameter {param_type} {param} = {param_default}\n\n"
        docstring += f"        {desc}\n\n"

    for port, desc in doc.ports.items():
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
    docstring = ""
    docstring += "Verilog Modules\n"
    docstring += "===============\n\n"

    docstring += ".. toctree::\n"
    docstring += "    :maxdepth: 2\n"
    docstring += "    :caption: Contents:\n\n"

    for file in files:
        docstring += f"    {os.path.splitext(file)[0]}.rst\n"
    doc_dir = __file__.split("/")[:-2]
    doc_dir.extend(["source", "verilog"])
    output_filename = "/".join(doc_dir) + "/index.rst"
    with open(output_filename, "w") as f_output:
        f_output.write(docstring)

if __name__ == "__main__":
    hdl_dir = __file__.split("/")[:-3]
    hdl_dir.extend(["hdl"])
    hdl_dir = "/".join(hdl_dir)
    generated_doc_files = []
    for root, dirs, files in os.walk(hdl_dir):
        for filename in sorted(files):
            if filename.endswith(".v") or filename.endswith(".v"):
                try: 
                    generate_rst_file(os.path.join(root, filename))
                    generate_svg_file(os.path.join(root, filename))
                    generated_doc_files.append(filename)
                except Exception as e:
                    warnings.warn(f"Error processing {filename}: {e}")
                    logger.debug(f"{e}")
                    continue

    generate_index_rst(generated_doc_files)