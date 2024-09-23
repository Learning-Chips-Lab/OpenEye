# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
from pyparsing import Suppress, Word, alphanums, OneOrMore, \
  Group, LineEnd, SkipTo, Optional, \
  ZeroOrMore, Combine, Literal, cppStyleComment, \
  Regex, oneOf, dbl_slash_comment


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

    # Define module name
    module_name = (
        "module " + identifier("module_name") + newline
    )

    # Parameter definition: parameter name and value
    ptype     = Optional( oneOf('integer real realtime time'), default='integer')
    param_def = Group((Suppress("parameter") | Suppress("localparam")) + ptype("p_type") + identifier("p_name") + Optional(Suppress("=")) + Optional(Combine(number)("p_value"), default=None) + Optional(Suppress(",")))

    parameters = Group(
        Suppress("#(") +
        OneOrMore(param_def, stopOn = ")") + 
        Suppress(")")
    )("parameters")

    # Port direction: input, output, inout
    port_dir = Literal("input") | Literal("output") | Literal("inout")
    
    # Port definition: direction and identifier, optionally with width [MSB:LSB]
    port_sign = Optional( oneOf('signed unsigned'), default='unsigned')
    port_type = Optional( oneOf('wire reg logic'), default='wire')
    port_range = Optional(Group(Suppress("[") + number("upper") + Suppress(":") + number("lower") + Suppress("]")), default=[[1, 0]])
    port_def = Group(port_dir("direction") + port_type("type") + port_sign("sign") + port_range("range") + identifier("name") + Optional(Suppress(",")))

    ports = Group(
        Suppress("(") +
        OneOrMore(port_def,  stopOn=");") +
        Suppress(");")
    )("ports")
    ports.ignore(dbl_slash_comment)

    # Module definition: module name, optional parameters, and ports
    module_block = (
        SkipTo(module_name) + module_name +
        Optional(parameters) +
        ports + 
        SkipTo("endmodule") + Suppress("endmodule")
    )

    comment = cppStyleComment
    module_block.ignore(comment)


    results = module_block.parseString(text, parseAll=False)

    return ModuleDefinition(
        module_name=results["module_name"],
        parameters={p["p_name"]: Parameter(p["p_name"], p["p_type"], p["p_value"]) for p in results.get("parameters", [])},
        ports={p["name"]: Port(p["name"], p["direction"], p["sign"], p["type"], p["range"]) for p in results.get("ports", [])},
        parser_results=results
    )
    
