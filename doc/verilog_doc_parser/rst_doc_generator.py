# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
"""Generate documented declarations and mandatory diagrams."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from textwrap import indent

if __package__:
    from .docstring_parser import MissingDocumentationError, parse_documentation
    from .moduledef_parser import parse_module_def, preprocess_header
else:
    from docstring_parser import MissingDocumentationError, parse_documentation
    from moduledef_parser import parse_module_def, preprocess_header


DEFINES = frozenset({"USE_INTERNAL_PARAMS", "USE_INTERNAL_PARAMS_PE", "USE_INTERNAL_PARAMS_PE_cluster"})


def declarations(doc, module):
    items = []
    for name, description in doc.parameters.items():
        if name in module.parameters:
            param = module.parameters[name]
            signature = " ".join(filter(None, (param.kind, param.ptype, name)))
            signature += f" = {param.default_value}"
            items.append({"kind": "parameter", "name": name, "signature": signature,
                          "description": description})
    for name, description in doc.ports.items():
        if name in module.ports:
            port = module.ports[name]
            signature = f"{port.direction} {port.type} {port.sign}"
            if port.packed_range is not None:
                signature += f" [{port.range_upper}:{port.range_lower}]"
            signature += f" {name}"
            items.append({"kind": "port", "name": name, "signature": signature,
                          "description": description})
    return items


def generate_rst(doc, module_def):
    ports = ", ".join(module_def.ports)
    text = f"{doc.module_name}\n{'=' * len(doc.module_name)}\n\n"
    text += f".. verilog:module:: module {doc.module_name}({ports})\n\n"
    text += indent(doc.description.replace("|", r"\|"), "    ") + "\n\n"
    text += f"    .. image:: __{doc.module_name}.png\n\n"
    for item in declarations(doc, module_def):
        text += f"    .. verilog:{item['kind']}:: {item['signature']}\n\n"
        text += indent(item["description"].replace("|", r"\|"), "        ") + "\n\n"
    return text


def check_png(path):
    from PIL import Image
    with Image.open(path) as picture:
        if picture.format != "PNG" or min(picture.size) < 2:
            raise ValueError(f"Invalid diagram: {path}")
        picture.load()


def generate(hdl_dir, output_dir, *, allow_partial=False, symbolator="symbolator"):
    hdl_dir, output_dir = Path(hdl_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {"profile": "structured-comments", "defines": sorted(DEFINES),
              "allow_partial": allow_partial, "diagrams": "required",
              "generated": [], "omitted": [], "failed": []}
    inputs = sorted(hdl_dir.rglob("*.v"))
    names = set()
    for source in inputs:
        entry = {"source": str(source.relative_to(hdl_dir))}
        try:
            source_bytes = source.read_bytes()
            entry["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
            text = source_bytes.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
            module = parse_module_def(text, defines=DEFINES)
            entry["module"] = module.module_name
            try:
                doc = parse_documentation(text)
            except MissingDocumentationError as error:
                entry["reason"] = str(error)
                report["omitted"].append(entry)
                continue
            if doc.module_name != module.module_name:
                raise ValueError(f"Comment module {doc.module_name} differs from declaration {module.module_name}")
            if module.module_name in names:
                raise ValueError(f"Duplicate module name: {module.module_name}")
            names.add(module.module_name)
            entry["ports"] = list(module.ports)
            entry["declarations"] = declarations(doc, module)
            entry["undocumented_parameters"] = sorted(module.parameters.keys() - doc.parameters.keys())
            entry["undocumented_ports"] = sorted(module.ports.keys() - doc.ports.keys())
            entry["unmatched_comment_parameters"] = sorted(doc.parameters.keys() - module.parameters.keys())
            entry["unmatched_comment_ports"] = sorted(doc.ports.keys() - module.ports.keys())
            entry["description"] = doc.description
            if module.module_name == "index":
                entry["page"] = "module-index"
            page = output_dir / f"{entry.get('page', module.module_name)}.rst"
            selected = output_dir / f"{module.module_name}.v"
            selected.write_text(preprocess_header(text, DEFINES))
            command = [symbolator, "-i", str(selected), "-t", "-o", str(output_dir) + "/", "-f", "PNG"]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            (output_dir / f"{module.module_name}.diagram.log").write_text(result.stdout)
            entry["diagram_command"] = command
            entry["diagram_exit_status"] = result.returncode
            if result.returncode:
                raise ValueError(f"Symbolator exited {result.returncode}; see {module.module_name}.diagram.log")
            check_png(output_dir / f"__{module.module_name}.png")
            page.write_text(generate_rst(doc, module), encoding="utf-8")
            report["generated"].append(entry)
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            entry["reason"] = f"{type(error).__name__}: {error}"
            report["failed"].append(entry)
    if not inputs:
        report["failed"].append({"source": str(hdl_dir), "reason": "No Verilog input files"})
    index = "Verilog Modules\n===============\n\n.. toctree::\n    :maxdepth: 2\n    :caption: Modules:\n\n"
    for entry in sorted(report["generated"], key=lambda item: item["module"]):
        index += f"    {entry.get('page', entry['module'])}\n"
    (output_dir / "index.rst").write_text(index)
    (output_dir / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdl-dir", type=Path, default=Path(__file__).resolve().parents[2] / "hdl")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--allow-partial", action="store_true", help="Allow missing structured comments; never allow parser or diagram failures")
    parser.add_argument("--symbolator", default="symbolator")
    args = parser.parse_args()
    try:
        report = generate(args.hdl_dir, args.output_dir, allow_partial=args.allow_partial, symbolator=args.symbolator)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Generation failed: {error}\n")
    for category in ("generated", "omitted", "failed"):
        print(f"{category}: {len(report[category])}")
        for entry in report[category]:
            print(f"  {entry['source']}: {entry.get('reason', entry.get('module', ''))}")
    return int(bool(report["failed"] or not report["generated"] or (report["omitted"] and not args.allow_partial)))


if __name__ == "__main__":
    raise SystemExit(main())