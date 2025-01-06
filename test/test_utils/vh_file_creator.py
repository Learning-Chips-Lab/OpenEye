
import sys
import os
directory = (os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
sys.path.extend([directory, os.path.dirname(os.path.realpath(__file__))])
import test_utils.open_eye_parameters as oep

def are_files_identical(file1_path, file2_path):
    try:
        with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
            file1_content = file1.read()
            file2_content = file2.read()

            return file1_content == file2_content
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"An unknown error occured: {e}")
        return False

serial = 0
openeye_parameter = oep.create_vh_file(serial, "pre_parameters.vh")
if (are_files_identical("pre_parameters.vh","parameters.vh")):
    os.remove("pre_parameters.vh")
    print("Same vh-file. Do not recompile")
else:
    toplevel = str(os.getenv("TOPLEVEL"))
    if (os.path.exists("parameters.vh")) :
        os.remove("parameters.vh")
    os.rename("pre_parameters.vh", "parameters.vh")
    print (os.getcwd() + "/../../hdl/" + toplevel + ".v")
    os.utime(os.getcwd() + "/../../hdl/" + toplevel + ".v", None)

    print("Different vh-file Same")

print("VH-File created")
