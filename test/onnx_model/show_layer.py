# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import h5py

with h5py.File("model_info.h5", "r") as hdf5_file:
    for group_name in hdf5_file.keys():
        group = hdf5_file[group_name]
        print("Group:", group_name)
    

