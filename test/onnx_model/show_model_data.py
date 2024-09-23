# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import h5py
with h5py.File("model_info.h5", "r") as hdf5_file:
    with open("model_info.txt", "w") as text_file:
        for group_name in hdf5_file.keys():
            text_file.write("Group: {}\n".format(group_name))
            group = hdf5_file[group_name]
            for dataset_name, dataset_value in group.items():
                text_file.write("Dataset: {}\n".format(dataset_name))  # Write the dataset name
                text_file.write("Data:\n")
                if isinstance(dataset_value, h5py.Dataset):  # Check if it's a dataset
                    data = dataset_value[()]  # Access dataset's value
                    if isinstance(data, bytes):
                        data = data.decode('utf-8')  # Decode bytes to string if necessary
                    text_file.write("{}\n\n".format(data))  # Write data to text file

