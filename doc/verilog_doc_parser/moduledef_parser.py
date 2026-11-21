# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
from pyparsing import Suppress, Word, alphanums, OneOrMore, \
  Group, LineEnd, SkipTo, Optional, \
  ZeroOrMore, Combine, Literal, cppStyleComment, \
  Regex, oneOf, dbl_slash_comment, Keyword, nested_expr, original_text_for, \
  dbl_quoted_string, delimitedList, ParseBaseException, Forward
import re

IDENTIFIER = r"[a-zA-Z_][a-zA-Z0-9_$]*"
DIRECTIVE = re.compile(r"^\s*`(\w+)\b(.*)$")


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
    def __init__(self, name, ptype=None, default_value=None, kind="parameter"):
        self.name = name
        self.ptype = ptype
        self.default_value = default_value
        self.kind = kind

class Port:
    def __init__(self, name, direction, sign=None, type=None, range=None):
        self.name = name
        self.direction = direction
        self.packed_range = tuple(range[0]) if range is not None else None
        if range is None:
            range = [[1, 0]]
        if not sign:
            self.sign = 'unsigned'
        else:
            self.sign = sign

        if not type:
            self.type = 'wire'
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


def _without_comments(text, mask_strings=False):
    """Preserve offsets and newlines, including when strings contain // or )."""
    pattern = r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/'

    def replace(match):
        value = match.group()
        if value.startswith('"') and not mask_strings:
            return value
        return re.sub(r"[^\n]", " ", value)

    return re.sub(pattern, replace, text)


def _header_bounds(text):
    masked = _without_comments(text, mask_strings=True)
    modules = list(re.finditer(r"\bmodule\b", masked))
    if len(modules) != 1:
        raise ValueError(f"Expected exactly one module per file; found {len(modules)}")
    endings = list(re.finditer(r"\bendmodule\b", masked))
    if len(endings) != 1 or endings[0].start() < modules[0].start():
        raise ValueError("Expected exactly one matching endmodule")
    start = modules[0].start()
    depth = 0
    for line in masked[:start].splitlines():
        match = DIRECTIVE.match(line)
        if match:
            if match[1] in ("ifdef", "ifndef"):
                depth += 1
            elif match[1] == "endif":
                depth -= 1
    if depth:
        raise ValueError("Conditionals enclosing a module declaration are unsupported")
    # A semicolon cannot occur in supported declaration types or defaults.
    # Strings/comments have already been masked.
    end = masked.find(";", start, endings[0].start())
    if end < 0:
        raise ValueError("Missing module header semicolon")
    return start, end + 1


def preprocess_header(text, defines=None):
    """Select header branches; return full source with its body unchanged."""
    start, end = _header_bounds(text)
    header = _without_comments(text[start:end])
    output, stack, active = [], [], True
    for line in header.splitlines(keepends=True):
        match = DIRECTIVE.match(line)
        if not match:
            if active:
                output.append(line)
            continue
        directive, argument = match[1], match[2].strip()
        if directive in ("ifdef", "ifndef"):
            if defines is None:
                raise ValueError("Conditional module header requires an explicit defines profile")
            if not re.fullmatch(IDENTIFIER, argument):
                raise ValueError(f"Unsupported `{directive} argument: {argument}")
            enabled = argument in defines
            if directive == "ifndef":
                enabled = not enabled
            stack.append((active, enabled, False))
            active = active and enabled
        elif directive == "else":
            if argument or not stack or stack[-1][2]:
                raise ValueError("Unexpected or duplicate `else in module header")
            parent, enabled, _ = stack[-1]
            stack[-1] = (parent, enabled, True)
            active = parent and not enabled
        elif directive == "endif":
            if argument or not stack:
                raise ValueError("Unexpected `endif in module header")
            active = stack.pop()[0]
        elif directive == "elsif" or active:
            raise ValueError(f"Unsupported `{directive} in module header")
    if stack:
        raise ValueError("Unterminated conditional in module header")
    selected = "".join(output)
    if "`" in _without_comments(selected, mask_strings=True):
        raise ValueError("Macro expansion in module header is unsupported")
    return text[:start] + selected + text[end:]


def parse_module_def(text, defines=None):
    # Define basic elements
    identifier = Regex(r"[a-zA-Z_][a-zA-Z0-9_\$]*")
    balanced = Forward()
    content = balanced | Regex(r'[^\s()\[\]{}"]+')
    balanced <<= (dbl_quoted_string
                  | nested_expr("(", ")", content=content, ignore_expr=dbl_quoted_string)
                  | nested_expr("[", "]", content=content, ignore_expr=dbl_quoted_string)
                  | nested_expr("{", "}", content=content, ignore_expr=dbl_quoted_string))
    grouped = original_text_for(balanced)
    number = original_text_for(OneOrMore(grouped | Regex(r'[^\s,()\[\]{}"]+')))
    newline = Suppress(LineEnd())
    bound = original_text_for(OneOrMore(grouped | Regex(r'[^\s,:()\[\]{}"]+')))
    packed_range = Group(Suppress("[") + bound("upper") + Suppress(":") + bound("lower") + Suppress("]"))

    # Strategy: Use a simple approach to extract module name, parameters, and ports
    # Match: module NAME [#(...)] (...) ;
    import re

    # Step 1: Find module name - ensure "module" is a keyword, not part of a comment
    # Look for "module" at the beginning of a line (after optional whitespace)
    text = preprocess_header(text, defines)
    start, end = _header_bounds(text)
    module_name_match = re.match(r'\s*module\s+([a-zA-Z_][a-zA-Z0-9_\$]*)', text[start:end])
    if not module_name_match:
        raise ValueError("Could not find module definition in text")

    module_name = module_name_match.group(1)
    search_start = start + module_name_match.end()
    # Count delimiters only outside strings/comments, preserving source offsets.
    masked = _without_comments(text, mask_strings=True)
    if '"' in dbl_quoted_string.suppress().transform_string(text[search_start:end]):
        raise ValueError("Unterminated string in module header")

    # Step 2: Look for #( to determine if there are parameters
    params_str = None
    ports_start_pos = search_start

    param_marker = re.match(r'\s*#\s*(?=\()', masked[search_start:end])
    if param_marker:
        param_start = search_start + param_marker.end() - 1
        # Found parameters - extract them
        # Find matching closing paren for #(...)
        paren_count = 0
        in_params = False
        param_end = -1
        for i in range(param_start + 1, end):
            if masked[i] == '(':
                paren_count += 1
                in_params = True
            elif masked[i] == ')':
                if in_params:
                    paren_count -= 1
                    if paren_count == 0:
                        param_end = i
                        break
                else:
                    in_params = False

        if param_end == -1:
            raise ValueError("Could not find parameter list closing parenthesis")
        params_str = text[param_start + 2:param_end]
        ports_start_pos = param_end + 1

    # Step 3: Extract ports (...)
    port_paren_start = masked.find('(', ports_start_pos, end)
    if port_paren_start == -1:
        raise ValueError("Could not find port list opening parenthesis")
    if text[ports_start_pos:port_paren_start].strip():
        raise ValueError("Unsupported text before ANSI port list")

    # Find matching closing paren
    paren_count = 0
    port_paren_end = -1
    for i in range(port_paren_start, end):
        if masked[i] == '(':
            paren_count += 1
        elif masked[i] == ')':
            paren_count -= 1
            if paren_count == 0:
                port_paren_end = i
                break

    if port_paren_end == -1:
        raise ValueError("Could not find port list closing parenthesis")
    if text[port_paren_end + 1:end].strip() != ';':
        raise ValueError("Unsupported text after ANSI port list")

    ports_str = text[port_paren_start + 1:port_paren_end]

    # Now parse the parameter and port strings separately
    # Parameter definition: parameter name and value
    ptype     = original_text_for(Optional(oneOf('integer real realtime time bit logic reg', asKeyword=True))
                                 + Optional(oneOf('signed unsigned', asKeyword=True)) + Optional(packed_range))
    param_def = Group((Keyword("parameter") | Keyword("localparam"))("kind") + ptype("p_type") + identifier("p_name") + Suppress("=") + number("p_value"))

    parameters_parser = Group(
        Optional(delimitedList(param_def))
    )("parameters")

    # Port direction: input, output, inout
    port_dir = Keyword("input") | Keyword("output") | Keyword("inout")

    # Port definition: direction and identifier, optionally with width [MSB:LSB]
    port_sign = Optional( oneOf('signed unsigned', asKeyword=True), default='unsigned')
    port_type = Optional( oneOf('wire reg logic', asKeyword=True), default='wire')
    port_range = Optional(Group(Suppress("[") + bound("upper") + Suppress(":") + bound("lower") + Suppress("]")))
    port_def = Group(port_dir("direction") + port_type("type") + port_sign("sign") + port_range("range") + identifier("name"))

    ports_parser = Group(
        Suppress("(") +
        Optional(delimitedList(port_def)) +
        Suppress(");")
    )("ports")
    ports_parser.ignore(dbl_slash_comment)

    # Parse parameters if present
    parsed_params = {}
    if params_str:
        try:
            params_results = parameters_parser.parseString(params_str, parseAll=True)
            param_list = params_results.get("parameters") or []
            parsed_params = {p["p_name"]: Parameter(p["p_name"], p["p_type"].strip(), p["p_value"].strip(), p["kind"]) for p in param_list}
        except ParseBaseException as error:
            raise ValueError(f"Unsupported parameter declaration: {error}") from error
        if len(parsed_params) != len(param_list):
            raise ValueError("Duplicate header parameter")

    # Parse ports - add the parentheses and semicolon to make it complete
    ports_text = "(" + ports_str + ");"
    try:
        ports_results = ports_parser.parseString(ports_text, parseAll=True)
    except ParseBaseException as error:
        remaining = ports_text[error.loc:].strip()
        context = "Unsupported ANSI port declaration" if error.loc <= 1 or remaining.startswith(",") else "Unsupported port name or trailing declaration"
        raise ValueError(f"{context}: {remaining.removesuffix(');').lstrip(',').strip()}") from error
    port_list = ports_results.get("ports") or []
    parsed_ports = {p["name"]: Port(p["name"], p["direction"], p["sign"], p["type"], p.get("range")) for p in port_list}
    if len(parsed_ports) != len(port_list):
        raise ValueError("Duplicate port")

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
    
