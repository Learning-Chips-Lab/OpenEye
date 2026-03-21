# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
from pyparsing import Suppress, Word, alphanums, OneOrMore, \
  Group, LineEnd, SkipTo, Optional, \
  ZeroOrMore, Combine, Literal, cppStyleComment, \
  Regex, oneOf, dbl_slash_comment, Keyword


class ModuleDefinition:
    """Class to store module definition information.

    Attributes:
        module_name (str): Name of the module.
        parameters (dict): Dictionary of parameters with name as key and Parameter object as value.
        ports (dict): Dictionary of ports with name as key and Port object as value.
        parser_results (dict): Dictionary of parser results.
    """
    def __init__(self, module_name, parameters=None, ports=None, parser_results=None):
        self.module_name = module_name
        self.parameters = parameters
        self.ports = ports
        self.parser_results = parser_results

class Parameter:
    """
    Class to store parameter information.

    Attributes:
        name (str): Name of the parameter.
        ptype (str): Type of the parameter (integer, real, realtime,).
        default_value (str): Default value of the parameter.
    """
    def __init__(self, name, ptype=None, default_value=None):
        self.name = name
        self.ptype = ptype
        self.default_value = default_value

class Port:
    def __init__(self, name, direction, sign=None, type=None, range=None):
        self.name = name
        self.direction = direction
        if not sign:
            sign = 'unsigned'
        else:
            self.sign = sign

        if not type:
            type = 'wire'
        else:
            self.type = type

        try:
            self.range_upper = int(range[0][0])
        except:
            self.range_upper = range[0][0]
        try:
            self.range_lower = int(range[0][1])
        except:
            self.range_lower = range[0][1]

def parse_module_def(text):
    # Define basic elements
    identifier = Regex(r"[a-zA-Z_][a-zA-Z0-9_\$]*")
    number = OneOrMore(Word(alphanums + r"_+-*/'\".\$() "), stopOn=LineEnd())  # Allow for parameter values to be expressions
    newline = Suppress(LineEnd())

    # Strategy: Use a simple approach to extract module name, parameters, and ports
    # Match: module NAME [#(...)] (...) ;
    import re

    # Step 1: Find module name - ensure "module" is a keyword, not part of a comment
    # Look for "module" at the beginning of a line (after optional whitespace)
    module_name_match = re.search(r'^\s*module\s+([a-zA-Z_][a-zA-Z0-9_\$]*)', text, re.MULTILINE)
    if not module_name_match:
        raise ValueError("Could not find module definition in text")

    module_name = module_name_match.group(1)
    search_start = module_name_match.end()

    # Step 2: Look for #( to determine if there are parameters
    params_str = None
    ports_start_pos = search_start

    param_start = text.find('#(', search_start)
    if param_start != -1 and param_start < text.find('(', search_start):
        # Found parameters - extract them
        # Find matching closing paren for #(...)
        paren_count = 0
        in_params = False
        param_end = -1
        for i in range(param_start + 1, len(text)):
            if text[i] == '(':
                paren_count += 1
                in_params = True
            elif text[i] == ')':
                if in_params:
                    paren_count -= 1
                    if paren_count == 0:
                        param_end = i
                        break
                else:
                    in_params = False

        if param_end != -1:
            params_str = text[param_start + 2:param_end]
            ports_start_pos = param_end + 1

    # Step 3: Extract ports (...)
    port_paren_start = text.find('(', ports_start_pos)
    if port_paren_start == -1:
        raise ValueError("Could not find port list opening parenthesis")

    # Find matching closing paren
    paren_count = 0
    port_paren_end = -1
    for i in range(port_paren_start, len(text)):
        if text[i] == '(':
            paren_count += 1
        elif text[i] == ')':
            paren_count -= 1
            if paren_count == 0:
                port_paren_end = i
                break

    if port_paren_end == -1:
        raise ValueError("Could not find port list closing parenthesis")

    ports_str = text[port_paren_start + 1:port_paren_end]

    # Now parse the parameter and port strings separately
    # Parameter definition: parameter name and value
    ptype     = Optional( oneOf('integer real realtime time'), default='integer')
    param_def = Group((Suppress("parameter") | Suppress("localparam")) + ptype("p_type") + identifier("p_name") + Optional(Suppress("=")) + Optional(Combine(number)("p_value"), default=None) + Optional(Suppress(",")))

    parameters_parser = Group(
        OneOrMore(param_def, stopOn = ")")
    )("parameters")

    # Port direction: input, output, inout
    port_dir = Literal("input") | Literal("output") | Literal("inout")

    # Port definition: direction and identifier, optionally with width [MSB:LSB]
    port_sign = Optional( oneOf('signed unsigned'), default='unsigned')
    port_type = Optional( oneOf('wire reg logic'), default='wire')
    port_range = Optional(Group(Suppress("[") + number("upper") + Suppress(":") + number("lower") + Suppress("]")), default=[[1, 0]])
    port_def = Group(port_dir("direction") + port_type("type") + port_sign("sign") + port_range("range") + identifier("name") + Optional(Suppress(",")))

    ports_parser = Group(
        Suppress("(") +
        OneOrMore(port_def,  stopOn=");") +
        Suppress(");")
    )("ports")
    ports_parser.ignore(dbl_slash_comment)

    # Parse parameters if present
    parsed_params = {}
    if params_str:
        try:
            params_results = parameters_parser.parseString(params_str, parseAll=False)
            param_list = params_results.get("parameters") or []
            parsed_params = {p["p_name"]: Parameter(p["p_name"], p["p_type"], p["p_value"]) for p in param_list}
        except Exception:
            # If parameter parsing fails, continue without parameters
            pass

    # Parse ports - add the parentheses and semicolon to make it complete
    # First, remove inline documentation comments (/// or //!) from ports_str
    ports_lines = ports_str.split('\n')
    filtered_ports_lines = []
    for line in ports_lines:
        # Remove everything after // or /// comments
        if '//' in line:
            line = line[:line.index('//')]
        filtered_ports_lines.append(line)
    ports_str_cleaned = '\n'.join(filtered_ports_lines)

    ports_text = "(" + ports_str_cleaned + ");"
    ports_results = ports_parser.parseString(ports_text, parseAll=False)
    port_list = ports_results.get("ports") or []
    parsed_ports = {p["name"]: Port(p["name"], p["direction"], p["sign"], p["type"], p["range"]) for p in port_list}

    # Create a simple result object
    class SimpleResults:
        def __init__(self):
            self.data = {
                "module_name": module_name,
                "parameters": parsed_params,
                "ports": parsed_ports
            }

        def get(self, key, default=None):
            return self.data.get(key, default)

        def __getitem__(self, key):
            return self.data[key]

    results = SimpleResults()

    return ModuleDefinition(
        module_name=results["module_name"],
        parameters=results["parameters"],
        ports=results["ports"],
        parser_results=results
    )
    
