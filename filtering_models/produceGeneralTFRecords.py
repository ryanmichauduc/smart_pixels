from pathlib import Path
import os, shutil
import glob
import time

import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir) 
import OptimizedDataGenerator4 as ODG

# Make general tf records directory
batch_size = 2000
directory_name = f'filtering_records{batch_size}test'
data_directory_path = "/home/elizahoward/MuonColliderSim/Simulation_Output/"
is_directory_recursive = False
file_type = "parquet"
data_format = "3D" # can't get 2D working
normalization = 1
file_fraction = .8 # fraction of files used for training
to_standardize = False
input_shape = (1,13, 21) # dimension of 3D cluster
transpose=(0, 2, 3, 1) # not sure what this does 
time_stamps=[19] # last timestamp is 19
x_feature_description = "all"
filteringBIB = True

records_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory_name)

if not os.path.exists(records_dir):
   os.mkdir(records_dir)
                
else:
    invalid = True
    while invalid:
        user_input = None
        while user_input != "overwrite" and user_input != "rename":
            user_input = input("Folder already in use, to overwite files, type \"overwrite\". To rename tf records directory, enter \"rename\"\n")

        if user_input == "overwrite":
            files = os.listdir(records_dir)
            for filename in files:
                file_path = os.path.join(records_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
            invalid = False

        else:
            stamp = input("Enter a new name: ")
            records_dir = Path(f"./{directory_name}").resolve()
            if not os.path.exists(records_dir):
                invalid = False
                os.mkdir(records_dir)

tf_dir_train = Path(records_dir, "tfrecords_train").resolve()
tf_dir_validation = Path(records_dir, "tfrecords_validation").resolve()

os.mkdir(tf_dir_train)
os.mkdir(tf_dir_validation)

total_files = len(glob.glob(
            data_directory_path + "recon" + data_format + "bib*." + file_type, 
            recursive=is_directory_recursive
            ))
file_count = round(file_fraction*total_files)

start_time = time.time()
training_generator = ODG.OptimizedDataGenerator(
                data_directory_path = data_directory_path,
                is_directory_recursive = is_directory_recursive,
                file_type = file_type,
                data_format = data_format,
                batch_size = batch_size,
                to_standardize= to_standardize,
                normalization=normalization,
                file_count=file_count,
                input_shape = input_shape,
                transpose = transpose,
                time_stamps = time_stamps,
                tf_records_dir = tf_dir_train,
                x_feature_description=x_feature_description,
                filteringBIB=filteringBIB,
            )
print("--- Training generator %s seconds ---" % (time.time() - start_time))

start_time = time.time()
validation_generator = ODG.OptimizedDataGenerator(
                data_directory_path = data_directory_path,
                is_directory_recursive = is_directory_recursive,
                file_type = file_type,
                data_format = data_format,
                batch_size = batch_size,
                to_standardize= to_standardize,
                normalization=normalization,
                file_count=total_files-file_count,
                files_from_end=True,
                input_shape = input_shape,
                transpose = transpose,
                time_stamps = time_stamps,
                tf_records_dir = tf_dir_validation,
                x_feature_description=x_feature_description,
                filteringBIB=filteringBIB,
            )
print("--- Validation generator %s seconds ---" % (time.time() - start_time))
