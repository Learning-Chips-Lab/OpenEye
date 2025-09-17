
import os
import sys
import test_utils.open_eye_parameters as oep
import test_utils.generic_test_utils as gtu

directory = (os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
sys.path.extend([directory, os.path.dirname(os.path.realpath(__file__))])
import test_utils.generic_test_utils as generic_test_utils

hdl_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir, "hdl")

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

def create_vh_file(openeye_parameter, filename = 'parameters.vh'):
    generic_test_utils.delete_files_in_directory('demo/')
    #os.makedirs(os.path.dirname(filename), exist_ok=True)
    txt_file = open(filename, 'w')
    txt_file.write("parameter CLUSTER_ROWS  = "  + str(openeye_parameter.Clusters_Y)+ ",\n")
    txt_file.write("parameter NUM_GLB_IACT  = "  + str(openeye_parameter.NUM_GLB_IACT)+ ",\n")
    txt_file.write("parameter PE_COLUMNS  = "  + str(openeye_parameter.PEs_X)+ ",\n")
    txt_file.write("parameter NUM_GLB_PSUM  = "  + str(openeye_parameter.NUM_GLB_PSUM)+ ",\n")
    txt_file.write("parameter NUM_GLB_WGHT = "  + str(openeye_parameter.NUM_GLB_WGHT)+ ",\n")
    txt_file.write("parameter PE_ROWS  = "  + str(openeye_parameter.PEs_Y)+ ",\n")
    txt_file.close()



def create_vh_file_from_envvars(file_path_vh = gtu.load_env_to_variable("VH_PATH", None), file_path_hdl = gtu.load_env_to_variable("HDL_PATH", hdl_dir), toplevel = gtu.load_env_to_variable("TOPLEVEL", None)):
    """Create VH file from environment variables."""
    file_path_vh = file_path_vh if file_path_vh else os.path.abspath(os.curdir)
    openeye_parameter = oep.get_oep(1)

    create_vh_file(openeye_parameter, file_path_vh + "/pre_parameters.vh")
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
        os.utime(file_path_hdl + "/" + toplevel + ".v", None)

        print("Different vh-file, updated vh-file")


if __name__ == "__main__":
    create_vh_file_from_envvars()

