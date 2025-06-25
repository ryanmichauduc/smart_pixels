# python imports
import tensorflow as tf
from qkeras import quantized_bits
from typing import Union, List, Tuple
import glob
import numpy as np
import pandas as pd
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import logging
import gc
import traceback
from pathlib import Path

# custom quantizer

# @tf.function
def QKeras_data_prep_quantizer(data, bits=4, int_bits=0, alpha=1):
    """
    Applies QKeras quantization.
    Args:
        data (tf.Tensor): Input data (tf.Tensor).
        bits (int): Number of bits for quantization.
        int_bits (int): Number of integer bits.
        alpha (float): (don't change)
    Returns::
        tf.Tensor: Quantized data (tf.Tensor).
    """
    quantizer = quantized_bits(bits, int_bits, alpha=alpha)
    return quantizer(data)


class OptimizedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, 
            data_directory_path: str = "./",
            is_directory_recursive: bool = False,
            file_type: str = "parquet",
            data_format: str = "3D",
            batch_size: int = 32,
            file_count = None,
            labels_list: Union[List,str] = None,
            to_standardize: bool = False,
            normalization: Union[list,int] = 1,
            input_shape: Tuple = (1, 13, 21),
            transpose = None,
            files_from_end = False,
            tag: str = "",
            x_feature_description: Union[list,str] = ['cluster'],
            filteringBIB: bool = True,

            # Added in Optimized datagenerators 
            load_records: bool = False,
            tf_records_dir: str = None,
            time_stamps = [19],
            quantize: bool = False,
            max_workers: int = 1,
            ):
        super().__init__() 

        """
        Data Generator to streamline data input to the network direct from the directory.
        Args:
        data_directory_path:
        labels_directory_path: 
        is_directory_recursive: 
        file_type: Default: "parquet"
                   Adapt the data loader according to file type. For now, it only supports csv and parquet file formats.
        data_format: Default: 3D
                    Used to refer to the relevant "recon" files, 2D for 2D pixel array, 3D for time series input,  
                    I can't get 2D working, so I just take the last time slice of the 3D cluster
        batch_size: Default: 32
                    The no. of data points to be included in a single batch.
        file_count: Default: None
                    To limit the no. of .csv files to be used for training.
                    If set to None, all files will be considered as legitimate inputs.
        labels_list: Default: "cotAlpha"
                     Input column name or list of column names to be used as label input to the neural network.
        x_feature_description: Default: ['cluster']
                    Which features do you want to save in (or load from) the tf record.   
                    If loading from tf records, you don't have to load every feature that was saved, you can just load the ones you want
        to_standardize: If set to True, it ensures that batches are normalized prior to being used as inputs
                        for training.
                        Default: False
        input_shape: Default: (1,13, 21) for image input to a 3D feedforward neural network.
                    To reshape the input array per the requirements of the network training.
        current: Default False, calculate the current instead of the integrated charge
        sample_delta_t: how long an "ADC bin" is in picoseconds
        
        load_from_training_dir: Directory to load prepared data from TFRecords.
        training_dir: Directory to save TFRecords.
        time_stamps: which of the 20 time stamps to train on. default -1 is to train on all of them
        seed: Random seed for shuffling.
        quantize: Whether to quantize the data.
        """
        if tf_records_dir is None:
            raise ValueError(f"tf_records_dir is None")

        self.file_offsets = [0] # this is leftover from the originial version and I don't know what it is for
        
        # This is the list of features that the code is currently built to handle
        allowed_features = ['cluster', 'x_profile', 'y_profile', 'x_size', 'y_size', 'y_local', 'z_global', 'total_charge', 'adjusted_hit_time', 'adjusted_hit_time_30ps_gaussian', 'adjusted_hit_time_60ps_gaussian']
        
        # you can just send in "all" and it will use the full list of allowed features
        if isinstance(x_feature_description, str) and x_feature_description == "all":
            self.x_feature_description=allowed_features

        elif isinstance(x_feature_description, str): raise Exception("x_feature_description must be a list of features or \'all\'")
        
        # check that the listed features are allowed
        else:
            invalid_items = [item for item in x_feature_description if item not in allowed_features]
            
            if invalid_items:
                raise Exception(f"The following features are not allowed in x_feature_description: {invalid_items}\nAllowed features include: {allowed_features}")
            self.x_feature_description = x_feature_description

        # If data is already prepared load and use that data
        if load_records:
            if not os.path.isdir(tf_records_dir):
                raise ValueError(f"Directory {tf_records_dir} does not exist.")
            else:
                self.tf_records_dir = tf_records_dir
        
        # otherwise write the tf record
        else:
            self.normalization = normalization

            # decide on which time stamps to load
            self.time_stamps = np.arange(0,20) if time_stamps == -1 else time_stamps
            len_xy, ntime = 13*21, 20
            idx = [[i*(len_xy),(i+1)*(len_xy)] for i in range(ntime)] # 20 time stamps of length 13*21
            self.time_stamps = np.array([ np.arange(idx[i][0], idx[i][1]).astype("str") for i in self.time_stamps]).flatten().tolist()
            if time_stamps != -1 and data_format != '2D':
                assert len(time_stamps) == input_shape[0]

            self.max_workers = max_workers
            
            if file_type not in ["csv", "parquet"]:
                raise ValueError("file_type can only be \"csv\" or \"parquet\"!")
            self.file_type = file_type

            # pull all of the bib files
            self.recon_files_bib = [
                f for f in glob.glob(
                    data_directory_path + "recon" + data_format + "bib*." + file_type, 
                    recursive=is_directory_recursive
                ) if tag in f
            ]
            # get all of the sig files
            self.recon_files_sig = [
                f for f in glob.glob(
                    data_directory_path + "recon" + data_format + "sig*." + file_type, 
                    recursive=is_directory_recursive
                ) if tag in f
            ]
            self.recon_files_sig.sort()
            self.recon_files_bib.sort()
            
            # Get a subset of the files for validation (usually use files_from_end) or training making sure that you don't overlap
            if file_count != None:
                if not files_from_end:
                    self.recon_files = self.recon_files_bib[:file_count]+self.recon_files_sig[:file_count]
                else:
                    self.recon_files = self.recon_files_bib[-file_count:]+self.recon_files_sig[-file_count:]
            else:
                self.recon_files = self.recon_files_bib+self.recon_files_sig

            self.dataset_mean = None
            self.dataset_std = None

            self.batch_size = batch_size
            self.labels_list = labels_list
            self.input_shape = input_shape
            self.transpose = transpose
            self.to_standardize = to_standardize
            
            labels_df = pd.DataFrame()
            recon_df = pd.DataFrame()
            ylocal_df = pd.DataFrame()
            z_loc_df = pd.DataFrame()
            eh_pairs = pd.DataFrame()
            hit_time_df = pd.DataFrame()
            hit_time_30_df = pd.DataFrame()
            hit_time_60_df = pd.DataFrame()

            for file in self.recon_files:
                tempDf = pd.read_parquet(file, columns=self.time_stamps)
                recon_df = pd.concat([recon_df,tempDf])

                # Swap recon3D for labels in the file name to get the corresponding labels file
                file = file.replace(f"recon{data_format}","labels")
                
                if not filteringBIB: 
                    # If you aren't filtering BIB for some reason, you can get truth info from the labels file
                    labels_df = pd.concat([labels_df,pd.read_parquet(file, columns=self.labels_list)])
                else:
                    if "sig" in file:
                        labels_df = pd.concat([labels_df, pd.DataFrame({'signal': [1] * tempDf.shape[0]})])
                    else:
                        labels_df = pd.concat([labels_df, pd.DataFrame({'signal': [0] * tempDf.shape[0]})])

                ylocal_df = pd.concat([ylocal_df,pd.read_parquet(file, columns=['y-local'])])
                eh_pairs = pd.concat([eh_pairs,pd.read_parquet(file, columns=['number_eh_pairs'])])
                z_loc_df = pd.concat([z_loc_df,pd.read_parquet(file, columns=['z-global'])])
                hit_time_df = pd.concat([hit_time_df,pd.read_parquet(file, columns=['adjusted_hit_time'])])
                hit_time_30_df = pd.concat([hit_time_30_df,pd.read_parquet(file, columns=['adjusted_hit_time_30ps_gaussian'])])
                hit_time_60_df = pd.concat([hit_time_60_df,pd.read_parquet(file, columns=['adjusted_hit_time_60ps_gaussian'])])

            # Get rid of any invalid values (I don't think there should ever be any, but just in case)
            has_nans = np.any(np.isnan(recon_df.values), axis=1)
            has_nans = np.arange(recon_df.shape[0])[has_nans]

            recon_df_raw = recon_df.drop(has_nans)
            labels_df_raw = labels_df.drop(has_nans)
            ylocal_df_raw = ylocal_df.drop(has_nans)
            z_loc_df_raw = z_loc_df.drop(has_nans)
            eh_pairs_raw = eh_pairs.drop(has_nans)
            hit_time = hit_time_df.drop(has_nans).values
            hit_time_30 = hit_time_30_df.drop(has_nans).values
            hit_time_60 = hit_time_60_df.drop(has_nans).values

            self.dataPoints = len(labels_df_raw)

            # Log normalization to avoid compressing small numbers to zero and to make difference between small numbers visible
            recon_values = recon_df_raw.values    
            nonzeros = abs(recon_values) > 0
            recon_values[nonzeros] = np.sign(recon_values[nonzeros])*np.log1p(abs(recon_values[nonzeros]))/math.log(2)
            
            if self.to_standardize:
                recon_values[nonzeros] = self.standardize(recon_values[nonzeros])
            
            recon_values = recon_values.reshape((-1, *self.input_shape))            
                        
            if self.transpose is not None:
                recon_values = recon_values.transpose(self.transpose)
            
            clusters = recon_values
            
            if time_stamps is list and len(time_stamps) == 1:
                clusters = recon_values.reshape((recon_values.shape[0],13,21))
            
            #  Get x and y profiles
            # make sure summing on right axis
            y_profiles = np.sum(clusters, axis = 2) 
            x_profiles = np.sum(clusters, axis = 1)

            # Get x and y sizes
            bool_arr = x_profiles != 0
            x_sizes = np.sum(bool_arr, axis = 1)/21 
            bool_arr = y_profiles != 0
            y_sizes = np.sum(bool_arr, axis = 1)/13
            
            # scale values to range between 0 and 1
            y_locals = ylocal_df_raw.values/8.5
            z_locs = z_loc_df_raw.values/65
            eh_pairs = eh_pairs_raw.values/150000 # Scale better here

            # This does nothing. Remove
            y_profiles=y_profiles.reshape((-1,13))
            x_profiles=x_profiles.reshape((-1,21))

            # Save the labels 
            self.labels = labels_df_raw.values

            # Save the x features
            self.x_features = {}
            if 'cluster' in self.x_feature_description:
                self.x_features['cluster'] = clusters
            if 'x_profile' in self.x_feature_description:
                self.x_features['x_profile'] = x_profiles
            if 'y_profile' in self.x_feature_description:
                self.x_features['y_profile'] = y_profiles
            if 'x_size' in self.x_feature_description:
                self.x_features['x_size'] = x_sizes
            if 'y_size' in self.x_feature_description:
                self.x_features['y_size'] = y_sizes
            if 'y_local' in self.x_feature_description:
                self.x_features['y_local'] = y_locals
            if 'z_global' in self.x_feature_description:
                self.x_features['z_global'] = z_locs
            if 'total_charge' in self.x_feature_description:
                self.x_features['total_charge'] = eh_pairs
            if 'adjusted_hit_time' in self.x_feature_description:
                self.x_features['adjusted_hit_time'] = hit_time
            if 'adjusted_hit_time_30ps_gaussian' in self.x_feature_description:
                self.x_features['adjusted_hit_time_30ps_gaussian'] = hit_time_30
            if 'adjusted_hit_time_60ps_gaussian' in self.x_feature_description:
                self.x_features['adjusted_hit_time_60ps_gaussian'] = hit_time_60
                
            self.tf_records_dir = tf_records_dir

            os.makedirs(self.tf_records_dir, exist_ok=True)

            # Write x_feature names in text file for record keeping
            with open(f'{self.tf_records_dir}/x_features.txt', "w") as f:
                for x_feature in self.x_feature_description:
                    f.write(f"{x_feature} ")

            self.save_batches_parallel() # save all the batches

        self.tfrecord_filenames = np.sort(np.array(tf.io.gfile.glob(os.path.join(self.tf_records_dir, "*.tfrecord"))))
        self.quantize = quantize
        self.epoch_count = 0
        self.on_epoch_end()

    # I don't use this, because it caused some of the data to not be included for some reason
    def process_file_parallel(self):
        file_infos = [(afile, self.time_stamps, self.file_type, self.input_shape, self.transpose) for afile in self.recon_files]
        results = []
        with ProcessPoolExecutor(self.max_workers) as executor:
            futures = [executor.submit(self._process_file_single, file_info) for file_info in file_infos]
            for future in tqdm(as_completed(futures), total=len(file_infos), desc="Processing Files..."):
                results.append(future.result())

        for amean, avariance, amin, amax, num_rows in results:
            self.file_offsets.append(self.file_offsets[-1] + num_rows)

            if self.dataset_mean is None:
                self.dataset_max = amax
                self.dataset_min = amin
                self.dataset_mean = amean
                self.dataset_std = avariance
            else:
                self.dataset_max = max(self.dataset_max, amax)
                self.dataset_min = min(self.dataset_min, amin)
                self.dataset_mean += amean
                self.dataset_std += avariance

        self.dataset_mean = self.dataset_mean / len(self.recon_files)
        self.dataset_std = np.sqrt(self.dataset_std / len(self.recon_files)) 
            
        self.file_offsets = np.array(self.file_offsets)

    @staticmethod
    def _process_file_single(file_info):
        afile, time_stamps, file_type, input_shape, transpose = file_info
        if file_type == "csv":
            adf = pd.read_csv(afile).dropna()
        elif file_type == "parquet":
            adf = pd.read_parquet(afile, columns=time_stamps).dropna()
    
        x = adf.values
        nonzeros = abs(x) > 0
        x[nonzeros] = np.sign(x[nonzeros]) * np.log1p(abs(x[nonzeros])) / math.log(2)
        amean, avariance = np.mean(x[nonzeros], keepdims=True), np.var(x[nonzeros], keepdims=True) + 1e-10
        centered = np.zeros_like(x)
        centered[nonzeros] = (x[nonzeros] - amean) / np.sqrt(avariance)
        x = x.reshape((-1, *input_shape))
        if transpose is not None:
            x = x.transpose(transpose)
        amin, amax = np.min(centered), np.max(centered)
        len_adf = len(adf)
        del adf
        gc.collect()
        
        return amean, avariance, amin, amax, len_adf

    def standardize(self, x, norm_factor_pos=1.7, norm_factor_neg=2.5):
        """
        Applies the normalization configuration in-place to a batch of inputs.
        `x` is changed in-place since the function is mainly used internally
        to standardize images and feed them to your network.
        Args:
            x: Batch of inputs to be normalized.
        Returns:
            The inputs, normalized. 
        """
        out = (x - self.dataset_mean)/self.dataset_std
        out[out > 0] = out[out > 0]/norm_factor_pos
        out[out < 0] = out[out < 0]/norm_factor_neg
        return out

    def save_batches_parallel(self):
        """
        Saves data batches as multiple TFRecord files.
        """
        num_batches = round(math.ceil(self.labels.shape[0]/self.batch_size)) # Total num of batches
        paths_or_errors = []

        # The max_workers is set to 1 because processing large batches in multiple threads can significantly
        # increase RAM usage. Adjust 'max_workers' based on your system's RAM capacity and requirements.
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_batch = {executor.submit(self.save_single_batch, i): i for i in range(num_batches)}
            
            for future in tqdm(as_completed(future_to_batch), total=num_batches, desc="Saving batches as TFRecords"):
                result = future.result()
                paths_or_errors.append(result)
            
        for res in paths_or_errors:
            if "Error" in res:
                print(res)  
                
    def save_single_batch(self, batch_index):
        """
        Serializes and saves a single batch to a TFRecord file.
        Args:
            batch_index (int): Index of the batch to save.
        Returns:
            str: Path to the saved TFRecord file or an error message.
        """
        
        try:
            filename = f"batch_{batch_index}.tfrecord"
            TFRfile_path = os.path.join(self.tf_records_dir, filename)

            X, y = self.prepare_batch_data(batch_index)
            serialized_example = self.serialize_example(X,y)

            with tf.io.TFRecordWriter(TFRfile_path) as writer:
                writer.write(serialized_example)
            return TFRfile_path
        
        except Exception as e:
            tb = traceback.format_exc()
            return f"Error saving batch {batch_index}: {e} \n{tb}" 
    
    def prepare_batch_data(self, batch_index):
        """
        Used to fetch a batch of inputs (X,y) for the network's training.
        """
        index = batch_index * self.batch_size # absolute *event* index
        
        y = self.labels[index:index+self.batch_size] / self.normalization 

        X = []

        for x_feature in self.x_feature_description:
            X.append(self.x_features[x_feature][index:index+self.batch_size])

        return X, y

    
    def serialize_example(self, X, y):
        """
        Serializes a single example (featuresand labels) to TFRecord format. 
        
        Args:
        - X: Training data
        - y: labels
        
        Returns:
        - string (serialized TFRecord example).
        """

        # X and y are float32 (maybe we can reduce this
        y = tf.cast(y, tf.float32)

        feature = {
            'y': self._bytes_feature(tf.io.serialize_tensor(y)),
        }

        for x_feature, x_feature_name in zip(X, self.x_feature_description):
            x_feature = tf.cast(x_feature, tf.float32)
            feature[x_feature_name] = self._bytes_feature(tf.io.serialize_tensor(x_feature))

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    @staticmethod
    def _bytes_feature(value):
        """
        Converts a string/byte value into a Tf feature of bytes_list
        
        Args: 
        - string/byte value
        
        Returns:
        - tf.train.Feature object as a bytes_list containing the input value.
        """
        if isinstance(value, type(tf.constant(0))): # check if Tf tensor
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def __getitem__(self, batch_index):
        """
        Load the batch from a pre-saved TFRecord file instead of processing raw data.
        Each file contains exactly one batch.
        quantization is done here: Helpful for pretraining without the quantization and the later training with quantized data.
        shuffling is also done here.
        TODO: prefetching (un-done)
        """
                
        tfrecord_path = self.tfrecord_filenames[batch_index]
        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
        parsed_dataset = raw_dataset.map(self._parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        for data in parsed_dataset:
            ''' Add the reshaping in saving'''
            X_batch, y_batch = data

            y_batch = tf.reshape(y_batch, [-1, *y_batch.shape[1:]])
            
            for x_feature in X_batch.keys():
                X_batch[x_feature] = tf.reshape(X_batch[x_feature], [-1, *X_batch[x_feature].shape[1:]])
            
            return X_batch, y_batch
    
    def _parse_tfrecord_fn(self, example):
        """
        Parses a single TFRecord example.
        
        Returns:
        - X: as a float32 tensor.
        - y: as a float32 tensor.
        """
    
        feature_description = {
            'y': tf.io.FixedLenFeature([], tf.string),
        }

        for x_feature in self.x_feature_description:
            feature_description[x_feature] = tf.io.FixedLenFeature([], tf.string)

        example = tf.io.parse_single_example(example, feature_description)

        y = tf.io.parse_tensor(example['y'], out_type=tf.float32)

        X = {}
        for x_feature in self.x_feature_description:
             X[x_feature]= tf.io.parse_tensor(example[x_feature], out_type=tf.float32)

        return X, y


    def __len__(self):
        if len(self.file_offsets) != 1: # used when TFRecord files are created during initialization
            num_batches = self.file_offsets[-1] // self.batch_size
        else: # used during loading saved TFRecord files
            files=[f for f in os.listdir(self.tf_records_dir) if f.endswith(".tfrecord")]
            num_batches = len(files)
        return num_batches

    def on_epoch_end(self):
        '''
        This shuffles the file ordering so that it shuffles the ordering in which the TFRecord
        are loaded during the training for each epochs.
        '''
        gc.collect()
        self.epoch_count += 1
        # Log quantization status once
        if self.epoch_count == 1:
            logging.warning(f"Quantization is {self.quantize} in data generator. This may affect model performance.")