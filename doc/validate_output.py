"""Check required generated declarations and diagrams in real Sphinx HTML."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from docutils.nodes import make_id
from PIL import Image


class Node:
    def __init__(self, tag, attrs=(), parent=None):
        self.tag, self.attrs, self.parent = tag, dict(attrs), parent
        self.children = []

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.walk()

    def text(self, *, own_description=False):
        """Keep inline text joined; prose excludes nested declarations and separates blocks."""
        classes = self.attrs.get("class", "").split()
        if (self.tag in {"script", "style"} or "headerlink" in classes
                or (own_description and self.tag == "dl" and "verilog" in classes)):
            return ""
        text = "".join(child.text(own_description=own_description) if isinstance(child, Node) else child
                       for child in self.children)
        if own_description and self.tag in {"p", "div", "li", "dt", "dd", "br", "blockquote", "pre"}:
            return "\n" + text + "\n"
        return text


class Page(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.root = Node("root")
        self.current = self.root
        self.feed(path.read_text())
        self.close()
        self.nodes = list(self.root.walk())

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs, self.current)
        self.current.children.append(node)
        if tag not in self.VOID:
            self.current = node

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        node = self.current
        while node.parent is not None:
            if node.tag == tag:
                self.current = node.parent
                return
            node = node.parent

    def handle_data(self, data):
        self.current.children.append(data)

    def declaration(self, anchor, label, kind):
        matches = [node for node in self.nodes if node.attrs.get("id") == anchor]
        if len(matches) != 1 or matches[0].tag != "dt":
            raise ValueError(f"Missing or ambiguous domain anchor for {label}: {anchor}")
        signature = matches[0]
        if signature.parent.tag != "dl" or not {"verilog", kind}.issubset(signature.parent.attrs.get("class", "").split()):
            raise ValueError(f"Missing domain definition for {label}")
        bodies = [node for node in signature.parent.children if isinstance(node, Node) and node.tag == "dd"]
        if len(bodies) != 1:
            raise ValueError(f"Missing domain description for {label}")
        return signature, bodies[0]


def normalized(text):
    """Ignore RST inline/list markers and HTML typography, not prose words."""
    text = text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "--", "—": "---", "…": "..."}))
    text = re.sub(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+", "", text)
    text = re.sub(r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])", r"\1", text)
    text = re.sub(r"(?<!\w)\*{1,2}([^*\n]+)\*{1,2}(?!\w)", r"\1", text)
    return " ".join(text.replace("`", "").split())


def normalized_signature(text):
    """Ignore token spacing, preserving identifiers, operators and quoted strings."""
    tokens = re.findall(
        r'"(?:\\.|[^"\\])*"|\\\S+|[a-zA-Z0-9_$]+|'
        r'===|!==|<<<|>>>|==|!=|<=|>=|<<|>>|&&|\|\||\*\*|'
        r'\+\+|--|~&|~\||~\^|\^~|\+:|-:|[^\s]',
        text,
    )
    return tokens[:-1] if tokens[-1:] == [";"] else tokens


def require_description(expected, node, label):
    if normalized(expected) != normalized(node.text(own_description=True)):
        raise ValueError(f"Rendered description differs for {label}")


def local_image(source, page_path, html_dir):
    parts = urlsplit(source)
    if parts.scheme or parts.netloc:
        return None
    path = (html_dir / unquote(parts.path).lstrip("/")) if parts.path.startswith("/") else (page_path.parent / unquote(parts.path))
    path = path.resolve()
    if not path.is_relative_to(html_dir):
        raise ValueError(f"Image escapes HTML directory: {source}")
    if not path.is_file():
        raise ValueError(f"Missing referenced image: {source} in {page_path.name}")
    return path


def validate_output(generated_dir, html_dir):
    generated_dir, html_dir = Path(generated_dir).resolve(), Path(html_dir).resolve()
    coverage = json.loads((generated_dir / "coverage.json").read_text())
    entries = coverage.get("generated", [])
    if not entries or coverage.get("failed"):
        raise ValueError("Coverage has no generated modules or has generation failures")
    if coverage.get("omitted") and not coverage.get("allow_partial"):
        raise ValueError("Coverage omits modules without explicit partial coverage")
    for required in (html_dir / "index.html", html_dir / "verilog" / "index.html"):
        if not required.is_file():
            raise ValueError(f"Missing required HTML page: {required}")
    index = Page(html_dir / "verilog" / "index.html")
    links = {node.attrs.get("href") for node in index.nodes if node.tag == "a"}
    declaration_count = 0
    image_paths = set()
    for entry in entries:
        module = entry["module"]
        page_name = entry.get("page", module)
        page_path = html_dir / "verilog" / f"{page_name}.html"
        if not page_path.is_file():
            raise ValueError(f"Missing required module HTML: {module}")
        if f"{page_name}.html" not in links:
            raise ValueError(f"Module absent from generated HTML index: {module}")
        page = Page(page_path)
        module_anchor = make_id(f"verilog-{module}")
        signature, body = page.declaration(module_anchor, module, "module")
        actual = normalized_signature(signature.text())
        if "ports" in entry:
            expected = normalized_signature(f"module {module}({', '.join(entry['ports'])})")
            heading_matches = actual == expected
        else:
            expected = normalized_signature(f"module {module}(")
            heading_matches = actual[:len(expected)] == expected
        if not heading_matches:
            raise ValueError(f"Rendered module signature differs for {module}")
        require_description(entry["description"], body, module)
        for declaration in entry["declarations"]:
            label = f"{module}.{declaration['name']}"
            anchor = make_id(f"{module_anchor}-{declaration['name']}")
            signature, description = page.declaration(anchor, label, declaration["kind"])
            actual = normalized_signature(signature.text())
            if actual != normalized_signature(declaration["signature"]):
                raise ValueError(f"Rendered declaration signature differs for {label}: {signature.text()}")
            require_description(declaration["description"], description, label)
            declaration_count += 1
        expected_name = f"__{module}.png"
        diagram_paths = []
        for node in page.nodes:
            if node.tag != "img":
                continue
            source = node.attrs.get("src", "")
            path = local_image(source, page_path, html_dir)
            if path is None:
                continue
            image_paths.add(path)
            if path.name == expected_name:
                diagram_paths.append(path)
        if not diagram_paths:
            raise ValueError(f"Missing required diagram reference for {module}: {expected_name}")
        generated_diagram = generated_dir / expected_name
        for diagram in diagram_paths:
            with Image.open(diagram) as picture:
                if picture.format != "PNG" or min(picture.size) < 2:
                    raise ValueError(f"Invalid PNG diagram for {module}")
                picture.load()
            if hashlib.sha256(diagram.read_bytes()).digest() != hashlib.sha256(generated_diagram.read_bytes()).digest():
                raise ValueError(f"Rendered diagram differs from generated diagram for {module}")
    return {"modules": len(entries), "declarations": declaration_count,
            "diagrams": len(entries), "local_images": len(image_paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument("--html-dir", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = validate_output(args.generated_dir, args.html_dir)
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
    except (ValueError, OSError) as error:
        parser.exit(1, f"HTML validation failed: {error}\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
