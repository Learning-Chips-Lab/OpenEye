# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.


from pyparsing import Suppress, Word, alphanums, OneOrMore, ZeroOrMore, Group, LineEnd, SkipTo, restOfLine, Optional, White, Regex, NotAny


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
    word = Word(alphanums + "_.-`")
    # Port/parameter names: alphanums, underscore, brackets, special chars (but NOT colon/hyphen delimiters)
    port_param_name = Word(alphanums + "_`[]$/%,*+()")
    description_line = SkipTo(LineEnd())

    # brief description
    brief_description_line = (
        single_line_comment + word("brief") + newline
    )

    # Empty documentation line: just /// or //! with optional whitespace, but NOT followed by text
    # This ensures "/// Ports:" doesn't match as an empty line
    empty_documentation_line = single_line_comment + Optional(whitespace) + newline

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
    # Port/param items MUST have leading whitespace (to distinguish from section headers)
    # followed by a name, then optional whitespace, then : or -, then description
    param_port_item = Group(
        whitespace + port_param_name("name") + Optional(whitespace) + (colon | minus) + restOfLine("description")
    )

    # A port/param line is either:
    # - An empty comment line (just /// or //!) - try this first
    # - A comment followed by whitespace and a param_port_item
    #   (section headers like "Ports:" don't have leading whitespace so won't match)
    # Order matters: try empty line first because it's simpler and less likely to partially consume
    param_port_line = empty_documentation_line | (single_line_comment + param_port_item + newline)

    # Parameters: collect lines until we hit "Ports:"
    # Since param_port_line won't match "Ports:" (no leading whitespace),
    # ZeroOrMore will naturally stop there
    parameters = Group(
        Suppress(single_line_comment + "Parameters:" + newline) +
        ZeroOrMore(param_port_line)
    )("parameters")

    # Ports: collect port lines until we hit another section header or end of comment block
    # Since param_port_line won't match section headers (they lack leading whitespace),
    # ZeroOrMore will naturally stop
    ports = Group(
        Suppress(single_line_comment + "Ports:" + newline) +
        ZeroOrMore(param_port_line)
    )("ports")
    
    # Define the entire documentation block
    # The structure is: [anything] module_name description [parameters] [ports]
    doc_block = (
        SkipTo(module_name) +
        module_name +
        description +
        Optional(parameters) +
        Optional(SkipTo(ports) + ports)
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
