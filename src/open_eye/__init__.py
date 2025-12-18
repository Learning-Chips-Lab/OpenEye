# Generate some default path information (so that we can refer to files easily)
import os
import sys
directory = (os.path.abspath(os.getcwd()))
sys.path.extend([directory, os.path.dirname(os.path.realpath(__file__))])

test_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir, "test")
hdl_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir, "hdl")