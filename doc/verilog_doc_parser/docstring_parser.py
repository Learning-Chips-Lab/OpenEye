# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

import re
from pyparsing import Suppress, Word, alphanums, OneOrMore, ZeroOrMore, Group, LineEnd, SkipTo, restOfLine, Optional, White, Regex


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
    # Port/parameter names: Must be valid Verilog identifiers
    # Key constraint: Must either be ALL_UPPERCASE (with underscores) or be all lowercase with underscores
    # Excludes: Section headers like "Ports", "Parameters", subsection headers like "Configuration"
    # Valid: DATA_WIDTH_SUM (uppercase with underscore), psum_ready_o (lowercase with underscore), clk_i (lowercase with underscore)
    # Invalid: Ports, Parameters, Configuration, Clock (section/subsection headers without underscores)
    # Pattern: Either ALL_CAPS with underscores, or starts with lowercase with underscores
    port_param_name = Regex(r'(?:[A-Z][A-Z0-9_]*_[A-Z0-9_]*|[a-z_][a-z0-9_]*)[`\[\]$/%,*+()]*')
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

    # Define description - capture everything until Parameters: or Ports:
    # This handles multi-paragraph descriptions with sections like "Overview:", "Architecture:", etc.
    description = Group(
        OneOrMore(
            (single_line_comment + description_line + newline),
            stopOn= single_line_comment + "Parameters:" | single_line_comment + "Ports:"
        )
    )("description")

    # Define parameters and ports
    # Port/param items MUST have leading whitespace (to distinguish from section headers)
    # followed by a name, then optional whitespace, then : or -, then description
    # Key constraint: The name must be a valid identifier (start with alphanumeric or underscore)
    # This filters out lines like "0:" or "1:" which are continuation lines
    param_port_item = Group(
        whitespace + port_param_name("name") + Optional(whitespace) + (colon | minus) + restOfLine("description")
    )

    # For parameters: include subsection headers/continuations but exclude "Ports:" line
    # Use a Regex that matches any text EXCEPT "Ports:"
    # This is tricky because we need to match "Ports" from different starting positions
    other_comment_not_ports = Suppress(
        single_line_comment +
        ~Regex(r'\s*Ports\s*:', re.IGNORECASE) +  # Negative match for "Ports:"
        SkipTo(LineEnd()) +
        newline
    )

    param_line_with_catch = (
        empty_documentation_line |
        (single_line_comment + param_port_item + newline) |
        other_comment_not_ports
    )

    # For ports section, include regular catch-all
    other_comment_line = Suppress(single_line_comment + SkipTo(LineEnd()) + newline)

    port_line = (
        empty_documentation_line |
        (single_line_comment + param_port_item + newline) |
        other_comment_line
    )

    # Parameters: collect with catch-all that excludes "Ports:"
    parameters = Group(
        Suppress(single_line_comment + "Parameters:" + newline) +
        ZeroOrMore(param_line_with_catch)
    )("parameters")

    # Ports: collect port lines including subsection headers and continuations
    ports = Group(
        Suppress(single_line_comment + "Ports:" + newline) +
        ZeroOrMore(port_line)
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
    # Filter out entries without "name" field (those are subsection headers and empty lines)
    return ModuleDocumentation(
        name=results["name"],
        description="\n".join(line.strip() for line in results["description"]).strip(),
        parameters={p["name"]: p["description"].strip() for p in results.get("parameters", []) if "name" in p},
        ports={p["name"]: p["description"].strip() for p in results.get("ports", []) if "name" in p},
        results=results
    )
