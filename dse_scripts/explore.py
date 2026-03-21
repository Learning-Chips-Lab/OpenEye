import subprocess
import os
import multiprocessing
import datetime
import shutil
import json
import open_eye.generator as gen
import raytune

PROJECT_XPR = "/home/lebold/work/LCL/fpga/mt_open_eye_zu19eg/mt_open_eye_zu19eg.xpr"
BLOCK_DESIGN = "/home/lebold/work/LCL/fpga/mt_open_eye_zu19eg/mt_open_eye_zu19eg.srcs/sources_1/bd/open_eye_bd/open_eye_bd.bd"
IP_NAME = "open_eye_v1_1_0"

# Generic parameter space
param_sets = [
    {"CLUSTER_ROWS": 2, "NUM_GLB_WGHT": 3, "NUM_GLB_PSUM": 4, "NUM_GLB_IACT": 3},
    {"CLUSTER_ROWS": 4, "NUM_GLB_WGHT": 3, "NUM_GLB_PSUM": 4, "NUM_GLB_IACT": 3},
    {"CLUSTER_ROWS": 6, "NUM_GLB_WGHT": 3, "NUM_GLB_PSUM": 4, "NUM_GLB_IACT": 3},
    {"CLUSTER_ROWS": 8, "NUM_GLB_WGHT": 3, "NUM_GLB_PSUM": 4, "NUM_GLB_IACT": 3},
    {"CLUSTER_ROWS": 2, "NUM_GLB_WGHT": 3, "NUM_GLB_PSUM": 2, "NUM_GLB_IACT": 3},
    {"CLUSTER_ROWS": 4, "NUM_GLB_WGHT": 3, "NUM_GLB_PSUM": 2, "NUM_GLB_IACT": 3},
    {"CLUSTER_ROWS": 6, "NUM_GLB_WGHT": 3, "NUM_GLB_PSUM": 2, "NUM_GLB_IACT": 3},
    {"CLUSTER_ROWS": 8, "NUM_GLB_WGHT": 3, "NUM_GLB_PSUM": 2, "NUM_GLB_IACT": 3},
    {"CLUSTER_ROWS": 2, "NUM_GLB_WGHT": 4, "NUM_GLB_PSUM": 4, "NUM_GLB_IACT": 3},
    {"CLUSTER_ROWS": 4, "NUM_GLB_WGHT": 4, "NUM_GLB_PSUM": 4, "NUM_GLB_IACT": 3},
    {"CLUSTER_ROWS": 6, "NUM_GLB_WGHT": 4, "NUM_GLB_PSUM": 4, "NUM_GLB_IACT": 3},
    {"CLUSTER_ROWS": 8, "NUM_GLB_WGHT": 4, "NUM_GLB_PSUM": 4, "NUM_GLB_IACT": 3},
]

def run_vivado(args):
    idx, params = args
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"run_{idx}_{timestamp}"
    report_dir = os.path.join(run_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)

    # Metadaten abspeichern
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # Projekt klonen, damit Runs sich nicht in die Quere kommen
    project_copy = os.path.join(run_dir, "project")
    shutil.copytree(os.path.dirname(PROJECT_XPR), project_copy)
    # Deklarierung des IP Repors in Projekt
    local_ip_repo = os.path.join(run_dir, "ip_repo")
    actual_ip_dir = os.path.join(local_ip_repo, "open_eye_mt_1.0")
    # Wir kopieren das aktuelle System-Environment, damit Vivado Pfade etc. behält
    custom_env = os.environ.copy()
    # Kopieren in das IP Repo
    original_ip_path = "/home/lebold/work/LCL/fpga/ip_repo/open_eye_mt_1.0"
    shutil.copytree(original_ip_path, actual_ip_dir)
    # Pfad definieren, wo die neue Verilog-Datei hin soll
    # Am besten direkt in den kopierten Projektordner
    generated_v_path = os.path.join(project_copy)
    generated_vh_path = os.path.join(project_copy)
    generated_py_path = os.path.join(project_copy)

    # --- Umgebungsvariablen für DIESEN Worker setzen ---
    for key, value in params.items():
        os.environ[str(key)] = str(value)

    # Generator aufrufen
    # Du musst dein generator.py entsprechend importieren oder via subprocess rufen
    gen.create_regmap_params_vh_file(".", generated_vh_path, generated_v_path, generated_py_path)

    target_v_in_repo = os.path.join(actual_ip_dir, "src", "dma_storage.v")
    shutil.copy(generated_v_path + "/dma_storage.v", target_v_in_repo)

    project_xpr_copy = os.path.join(project_copy, os.path.basename(PROJECT_XPR))
    bd_copy = os.path.join(project_copy, os.path.relpath(BLOCK_DESIGN, os.path.dirname(PROJECT_XPR)))

    # Parameter als String (key=value) übergeben
    param_strs = [f"{k}={v}" for k, v in params.items()]
    subprocess.run([
        "vivado", "-mode", "batch", 
        "-source", "run_impl.tcl",
        "-tclargs", project_xpr_copy, bd_copy, IP_NAME, report_dir,
        actual_ip_dir, *param_strs
    ], check=True)
    return f"Run {idx} finished, results in {report_dir}"

if __name__ == "__main__":
    with multiprocessing.Pool(processes=12) as pool:  # Amount = Licences
        results = pool.map(run_vivado, list(enumerate(param_sets)))

    for r in results:
        print(r)

