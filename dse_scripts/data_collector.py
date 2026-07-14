import os
import xml.etree.ElementTree as ET
import pandas as pd
import re
import json  # neu: fürs Einlesen der Parameter

def is_valid_resource(name: str) -> bool:
    """Filtert ungültige Ressourcennamen (Zahlen, leer, Yes/No)."""
    if not name or name.strip() == "":
        return False
    if re.fullmatch(r"\d+", name.strip()):  # nur Zahl
        return False
    if name.strip() in ["Yes", "No"]:
        return False
    return True

def parse_utilization_xml(xml_path):
    """Parst utilization.xml und gibt dict mit Ressourcennamen → Werte zurück."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    data = {}
    for section in root.findall(".//section"):
        for table in section.findall("table"):
            for tablerow in table.findall("tablerow"):
                cells = tablerow.findall("tablecell")
                if not cells:
                    continue

                resource = cells[0].get("contents", "").strip()
                if not is_valid_resource(resource):
                    continue

                used = cells[1].get("contents", "").strip() if len(cells) > 1 else ""
                avail = cells[4].get("contents", "").strip() if len(cells) > 4 else ""
                utilp = cells[5].get("contents", "").strip() if len(cells) > 5 else ""

                data[f"{resource} Used"] = used
                data[f"{resource} Available"] = avail
                data[f"{resource} Util%"] = utilp
    return data

def collect_all_runs(base_dir=".", out_csv="all_utilization.csv"):
    rows = []
    for run_dir in sorted(os.listdir(base_dir)):
        run_path = os.path.join(base_dir, run_dir)
        util_path = os.path.join(run_path, "reports", "utilization.xml")
        param_path = os.path.join(run_path, "params.json")

        if not os.path.isfile(util_path):
            continue

        row = {"Run": run_dir}

        # --- Parameter hinzufügen ---
        if os.path.isfile(param_path):
            with open(param_path, "r") as f:
                params = json.load(f)
            for k, v in params.items():
                row[k] = v

        # --- Utilization hinzufügen ---
        row.update(parse_utilization_xml(util_path))
        rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

if __name__ == "__main__":
    collect_all_runs(base_dir=".")

