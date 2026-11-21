# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

import re
from pyparsing import Suppress, Word, alphanums, OneOrMore, ZeroOrMore, Group, LineEnd, SkipTo, restOfLine, Optional, White, Regex, matchPreviousLiteral, ParseBaseException


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

class MissingDocumentationError(ValueError):
    """The source has no structured /// or //! Module: documentation block."""

def parse_documentation(text):
    # Define basic elements
    colon = Suppress(":")
    minus = Suppress("-")
    newline = Suppress(LineEnd())
    whitespace = Suppress(White(" \t"))

    # Define comment starters
    single_line_comment = Suppress("///") | Suppress("//!")

    # Define content elements
    word = Word(alphanums + "_.-`")
    # Port/parameter names: valid Verilog identifiers, including simple capitals
    # and mixed case. Section headers are excluded by the section rules below.
    port_param_name = Regex(r'[a-zA-Z_][a-zA-Z0-9_$]*[`\[\]$/%,*+()]*')
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
        Regex(r'[a-zA-Z_][a-zA-Z0-9_$]*')("name") + newline
    )

    # Define description - capture everything until Parameters: or Ports:
    # This handles multi-paragraph descriptions with sections like "Overview:", "Architecture:", etc.
    description = Group(
        ZeroOrMore(
            (single_line_comment + description_line + newline),
            stopOn= single_line_comment + "Parameters:" | single_line_comment + "Ports:"
        )
    )("description")

    # Define parameters and ports
    # Port/param items MUST have leading whitespace (to distinguish from section headers)
    # followed by a name, then optional whitespace, then : or -, then description
    # Key constraint: The name must be a valid identifier (start with alphanumeric or underscore)
    # This filters out lines like "0:" or "1:" which are continuation lines
    item_indentation = White(" \t", min=2).leaveWhitespace()
    continuation_indentation = matchPreviousLiteral(item_indentation).leaveWhitespace()
    param_port_item = Group(
        Suppress(item_indentation) + port_param_name("name") + Optional(whitespace) + (colon | minus) +
        (restOfLine.copy().leaveWhitespace().addCondition(lambda tokens: bool(tokens[0].strip())) + ZeroOrMore(
            newline + ~(single_line_comment + Regex(r'(?:Parameters|Ports):')) + single_line_comment +
            Suppress(continuation_indentation) + whitespace + Regex(r'\S[^\r\n]*').leaveWhitespace()
        )).setParseAction(lambda tokens: "\n".join(line.strip() for line in tokens))("description")
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

    param_line_with_catch = ~(single_line_comment + "Ports:") + (
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
        Optional(ports)
    )

    # Parse the text
    marker = re.search(r'(?m)^[ \t]*(?:///|//!)[ \t]*Module[ \t]*:', text)
    if not marker:
        raise MissingDocumentationError("No structured /// or //! Module: comment")
    block = re.match(r'(?:[ \t]*(?:///|//!)[^\n]*(?:\n|$))+', text[marker.start():]).group()
    try:
        module_name.parseString(block, parseAll=False)
    except ParseBaseException as error:
        raise ValueError("Invalid structured Module: name") from error
    text = text[:marker.start() + len(block)]
    try:
        results = doc_block.parseString(text, parseAll=False)
    except ParseBaseException as error:
        raise ValueError(f"Invalid structured documentation: {error}") from error

    # Create and return ModuleDocumentation object
    # Filter out entries without "name" field (those are subsection headers and empty lines)
    return ModuleDocumentation(
        name=results["name"],
        description="\n".join(line.strip() for line in results["description"]).strip(),
        parameters={p["name"]: p["description"].strip() for p in results.get("parameters", []) if "name" in p},
        ports={p["name"]: p["description"].strip() for p in results.get("ports", []) if "name" in p},
        results=results
    )
