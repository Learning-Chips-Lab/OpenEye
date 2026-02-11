import pathlib

def merge_sim_files():
    # Pfade definieren
    source_root = pathlib.Path(".temp")
    target_root = pathlib.Path("merged_results")

    # Zielordner erstellen, falls er nicht existiert
    target_root.mkdir(exist_ok=True)

    if not source_root.exists():
        print(f"Fehler: Der Ordner '{source_root}' wurde nicht gefunden.")
        return

    # Alle Unterordner in .temp durchlaufen
    for folder in source_root.iterdir():
        if folder.is_dir():
            param_file = folder / "parameters.vh"
            log_file = folder / "cocotb_sim.log"
            
            # Prüfen, ob beide Dateien existieren
            if param_file.exists() and log_file.exists():
                output_filename = target_root / f"{folder.name[64:-21]}.txt"
                
                try:
                    # Dateien lesen und zusammenführen
                    with open(param_file, 'r') as f1, open(log_file, 'r') as f2:
                        content1 = f1.read()
                        content2 = f2.read()
                    
                    with open(output_filename, 'w') as out:
                        #out.write(f"--- SOURCE FOLDER: {folder.name[64:-21]} ---\n")
                        out.write(f"--- START parameters.vh ---\n")
                        out.write(content1)
                        out.write(f"\n--- START cocotb_sim.log ---\n")
                        out.write(content2)
                    
                    print(f"Erfolgreich erstellt: {output_filename.name}")
                except Exception as e:
                    print(f"Fehler beim Verarbeiten von {folder.name}: {e}")
            else:
                print(f"Übersprungen: {folder.name} (Dateien fehlen)")

if __name__ == "__main__":
    merge_sim_files()