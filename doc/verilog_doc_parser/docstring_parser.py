# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.


from pyparsing import Suppress, Word, alphanums, OneOrMore, Group, LineEnd, SkipTo, restOfLine, Optional, White


class ModuleDocumentation:
    def __init__(self, name, brief_description=None, description=None, parameters=None, ports=None, results=None):
        self.module_name = name
        self.brief_description = brief_description
        self.description = description
        self.parameters = parameters
        self.ports = ports
        self.results = results

    @property
    def name(self):
        return self.module_name

def parse_documentation(text):
    # Define basic elements
    colon = Suppress(":")
    minus = Suppress("-")
    newline = Suppress(LineEnd())
    whitespace = Suppress(White())
    
    # Define comment starters
    single_line_comment = Suppress("///") | Suppress("//!")
    
    # Define content elements
    word = Word(alphanums + "_" + "." + "-" + "`")
    description_line = SkipTo(LineEnd())

    # brief description
    brief_description_line = (
        single_line_comment + word("brief") + newline
    )

    empty_documentation_line = single_line_comment + Optional(whitespace) + newline
    empty_documentation_line.debug = True
    
    # Define module name
    module_name = (
        single_line_comment + "Module:" + 
        word("name") + newline
    )
    
    # Define description
    description = Group(
        OneOrMore(
            (single_line_comment + description_line + newline),
            stopOn= single_line_comment + "Parameters:" | single_line_comment + "Ports:"
        )
    )("description")    
    
    # Define parameters and ports
    param_port_item = Group(
        word("name") + (colon | minus) + restOfLine("description")
    )
    parameters = Group(
        Suppress(single_line_comment + "Parameters:") +
        OneOrMore((single_line_comment + param_port_item + newline) | empty_documentation_line, 
                  stopOn = single_line_comment + "Ports:")
    )("parameters")
    
    ports = Group(
        Suppress(single_line_comment + "Ports:") +
        OneOrMore((single_line_comment + param_port_item + newline))
    )("ports")
    
    # Define the entire documentation block
    doc_block = (
        SkipTo(module_name) +
        module_name +
        description +
        Optional(parameters) +
        SkipTo(ports) + 
        Optional(ports)
    )
    
    # Parse the text
    results = doc_block.parseString(text, parseAll=False)

    # Create and return ModuleDocumentation object
    return ModuleDocumentation(
        name=results["name"],
        description="\n".join(line.strip() for line in results["description"]).strip(),
        parameters={p["name"]: p["description"].strip() for p in results.get("parameters", [])},
        ports={p["name"]: p["description"].strip() for p in results.get("ports", [])},
        results=results
    )
