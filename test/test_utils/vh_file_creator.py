
import sys
import os
import open_eye_parameters as oep

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

def create_vh_file(file_path_vh = os.getcwd(),file_path_hdl = os.getcwd() + "/../../hdl/", toplevel = None):
    serial = 0
    openeye_parameter = oep.create_vh_file(serial, file_path_vh + "/pre_parameters.vh")
    if (are_files_identical(file_path_vh + "/pre_parameters.vh",file_path_vh + "/parameters.vh")):
        os.remove(file_path_vh + "/pre_parameters.vh")
        print("Same vh-file. Do not recompile")
    else:
        if (toplevel == None):
            toplevel = str(os.getenv("TOPLEVEL"))
        if (os.path.exists(file_path_vh + "/parameters.vh")) :
            os.remove(file_path_vh + "/parameters.vh")
        os.rename(file_path_vh + "/pre_parameters.vh", file_path_vh + "/parameters.vh")
        print (file_path_hdl + toplevel + ".v")
        os.utime(file_path_hdl + toplevel + ".v", None)

        print("Different vh-file Same")

    print("VH-File created")

if __name__ == '__main__':
    create_vh_file()

