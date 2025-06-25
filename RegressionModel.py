# fix for keras v3.0 update
import os

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import roc_curve, auc

from qkeras import *
import OptimizedDataGenerator4 as ODG
import time

#os.environ['TF_USE_LEGACY_KERAS'] = '1' 

# python based
import random
from pathlib import Path
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
plt.rcParams['figure.dpi'] = 300
from tqdm import tqdm
from typing import Union
import glob
import numpy as np

# custom code
#from loss import *
from models import *

def scheduler(epoch, lr):
    if epoch == 100 or epoch == 130 or epoch == 160:
        return lr/2
    else:
        return lr
    
class RegressionModel:

    def __init__(self,
            data_directory_path: str = "./",
            is_directory_recursive: bool = False,
            file_type: str = "parquet",
            data_format: str = "3D",
            batch_size: int = 100,
            labels_list: list= ['x-midplane','y-midplane','cotAlpha','cotBeta'],
            units_list: list = ["[\u03BCm]", "[\u03BCm]", "", ""],
            normalization: Union[list,int] = np.array([75., 18.75, 8.0, 0.5]),
            muon_collider: bool = False,
            file_fraction: float = 0.8,
            to_standardize: bool = False,
            input_shape: tuple = (1,13,21),
            transpose = (0,2,3,1),
            use_time_stamps = [19],
            output_dir: str = "./ouput_prediction",
            learning_rate: float = 0.001,
            tag: str = "",
            x_feature_description: list = ['cluster'],
            filteringBIB: bool = False,
            training_dir: str = None,
            model_number: int = None,
            stamp: str = None,
            cut_hit_time: bool = False,
            hit_time_version: str = None
            ):

        if labels_list != None and len(labels_list) != 4:
            raise ValueError(f"Invalid list length: {len(labels_list)}. Required length is 4.")

        # Count total number of bib and sig files
        if muon_collider==True:
            total_files = len(glob.glob(
                    data_directory_path + "recon" + data_format + "bib*." + file_type, 
                    recursive=is_directory_recursive
                ))
            file_count = round(file_fraction*total_files)
        else:
            total_files = len(glob.glob(
                    data_directory_path + "recon" + data_format + f"{tag}*." + file_type, 
                    recursive=is_directory_recursive
                ))
            file_count = round(file_fraction*total_files)


        self.labels_list = labels_list
        self.units_list = units_list
        self.output_dir = output_dir
        self.normalization = normalization

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # create tf records directory
        
        if training_dir is None:
            use_tf_records = False
        else:
            use_tf_records = True

        self.history = None
        
        if not use_tf_records:
            if stamp is None:
                stamp = '%08x' % random.randrange(16**8)
            training_dir = Path(self.output_dir, f"training_records{stamp}").resolve()
            if not os.path.exists(training_dir):
                os.mkdir(training_dir)
            else: raise Exception("Folder already in use!")
            training_dir_train = Path(training_dir, "tfrecords_train").resolve()
            training_dir_validation = Path(training_dir, "tfrecords_test").resolve()
            os.mkdir(training_dir_train)
            os.mkdir(training_dir_validation)

        self.training_dir = f'{output_dir}/{training_dir}'

        start_time = time.time()
        if use_tf_records:
            self.training_generator = ODG.OptimizedDataGenerator(load_from_training_dir=self.training_dir, filteringBIB=filteringBIB, tf_records_dir='tfrecords_train')
        else:
            self.training_generator = ODG.OptimizedDataGenerator(
                data_directory_path = data_directory_path,
                is_directory_recursive = is_directory_recursive,
                file_type = file_type,
                data_format = data_format,
                muon_collider = muon_collider,
                batch_size = batch_size,
                to_standardize= to_standardize,
                normalization=normalization,
                file_count=file_count,
                labels_list = labels_list,
                input_shape = input_shape,
                transpose = transpose,
                use_time_stamps = use_time_stamps,
                training_dir = training_dir,
                tf_records_dir = "tfrecords_train",
                tag = tag,
                x_feature_description=x_feature_description,
                filteringBIB=filteringBIB,
                cut_hit_time=cut_hit_time,
                hit_time_version=hit_time_version
            )

        print("--- Training generator %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        if use_tf_records:
            self.validation_generator = ODG.OptimizedDataGenerator(load_from_training_dir=self.training_dir, tf_records_dir='tfrecords_test')
        else:
            self.validation_generator = ODG.OptimizedDataGenerator(
                data_directory_path = data_directory_path,
                is_directory_recursive = is_directory_recursive,
                file_type = file_type,
                data_format = data_format,
                muon_collider = muon_collider,
                batch_size = batch_size,
                to_standardize= to_standardize,
                normalization=normalization,
                file_count=total_files-file_count,
                files_from_end=True,
                labels_list = labels_list,
                input_shape = input_shape,
                transpose = transpose,
                use_time_stamps = use_time_stamps,
                training_dir = training_dir,
                tf_records_dir="tfrecords_test",
                tag = tag,
                x_feature_description=x_feature_description,
                filteringBIB=filteringBIB,
                cut_hit_time=cut_hit_time,
                hit_time_version=hit_time_version
            )

        print("--- Validation generator %s seconds ---" % (time.time() - start_time))

        self.n_filters = 5 # model number of filters
        self.pool_size = 3 # model pool size

        self.x_features=x_feature_description

        # Rearrange input shape for the model
        self.shape = (input_shape[1], input_shape[2], input_shape[0])

        if model_number is None:
            self.createModel()
        else:
            self.loadModel(model_number)

        self.compileModel(learning_rate=learning_rate)

        self.residuals = None

    def createModel(self):
        start_time = time.time()
        self.model=CreatePredictionModel(self.shape, self.n_filters, self.pool_size, self.include_y_local)
        self.model.summary()
        print("--- Model create and compile %s seconds ---" % (time.time() - start_time))

    def loadModel(self, model_number):
        self.modelName=f"Model {model_number}"
        file_path = Path(self.training_dir, f'Model_{model_number}.keras').resolve()
        self.model=tf.keras.models.load_model(file_path, compile=False)

    def saveModel(self, model_number, overwrite=False):
        file_path = Path(self.training_dir, f'Model_{model_number}.keras').resolve()
        if not overwrite and os.path.exists(file_path):
            raise Exception("Model exists. To overwrite existing saved model, set overwrite to True.")
        self.modelName=f"Model {model_number}"
        self.model.save(file_path)

    def compileModel(self, learning_rate):
        print(f"Compiling model with learning rate: {learning_rate}")
        self.learning_rate = learning_rate
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=custom_loss)

    def loadWeights(self, fileName):
        filePath = Path(self.training_dir, f'weights_{fileName}.hdf5').resolve()
        self.model.load_weights(filePath)

    def saveWeights(self, fileName):
        filePath = Path(self.training_dir, f'weights_{fileName}.hdf5').resolve()
        self.model.save_weights(filePath)

    def runTraining(self, epochs=50, early_stopping=True, save_all_weights=False, schedule_lr=True, save_prev_history=False):
        early_stopping_patience = 10

        # launch quick training once gpu is available
        es = EarlyStopping(
        patience=early_stopping_patience,
        restore_best_weights=True
        )
    
        # checkpoint path
        checkpoint_filepath = Path(self.output_dir,"weights", 'epoch.{epoch:02d}-t{loss:.2f}-v{val_loss:.2f}.weights.h5').resolve()
        mcp = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            save_best_only=True,
        )
        
        # learning rate scheduler
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

        callbacks = []
        if early_stopping:
            callbacks.append(es)
        if save_all_weights:
            callbacks.append(mcp)
        if schedule_lr:
            callbacks.append(lr_scheduler)

        # train
        prev_history = self.history

        self.history = self.model.fit(x=self.training_generator,
                        validation_data=self.validation_generator,
                        callbacks=callbacks,
                        epochs=epochs,
                        shuffle=False,
                        verbose=1)

        print(type(self.history))

        if save_prev_history and prev_history is not None:
            self.history += prev_history # check if this works ...

        self.residuals = None

        self.plotTraining()


    def plotTraining(self):
        fig, ax = plt.subplots(ncols=1, nrows=2, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax[0].plot(range(1,len(self.history.history['val_loss'])+1),self.history.history['val_loss'], c='royalblue')
        ax[0].set_ylabel("Validation Loss")
        ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        
        ax[1].plot(range(1,len(self.history.history['lr'])+1),self.history.history['lr'], c='seagreen')
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Learning Rate")
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[1].yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0e}'.format(x)))


        ax[0].grid(True, linestyle='--', linewidth=0.5, color='gray')
        ax[1].grid(True, linestyle='--', linewidth=0.5, color='gray')

        fig.tight_layout()

        plt.show()


    def checkResiduals(self):
        p_test = self.model.predict(self.validation_generator)

        complete_truth = None
        for _, y in tqdm(self.validation_generator):
            if complete_truth is None:
                complete_truth = y
            else:
                complete_truth = np.concatenate((complete_truth, y), axis=0)
        
        minval=1e-9

        # creates df with all predicted values and matrix elements - 4 predictions, all 10 unique matrix elements
        self.df = pd.DataFrame(p_test,columns=[self.labels_list[0],'M11',self.labels_list[1],'M22',self.labels_list[2],'M33',self.labels_list[3],'M44','M21','M31','M32','M41','M42','M43'])
        self.df[self.labels_list[0]] *= self.normalization[0]
        self.df[self.labels_list[1]] *= self.normalization[1]
        self.df[self.labels_list[2]] *= self.normalization[2]
        self.df[self.labels_list[3]] *= self.normalization[3]

        # stores all true values in same matrix as xtrue, ytrue, etc.
        self.df[self.labels_list[0]+'true'] = complete_truth[:,0]*self.normalization[0]
        self.df[self.labels_list[1]+'true'] = complete_truth[:,1]*self.normalization[1]
        self.df[self.labels_list[2]+'true'] = complete_truth[:,2]*self.normalization[2]
        self.df[self.labels_list[3]+'true'] = complete_truth[:,3]*self.normalization[3]
        self.df['M11'] = minval+tf.math.maximum(self.df['M11'], 0)
        self.df['M22'] = minval+tf.math.maximum(self.df['M22'], 0)
        self.df['M33'] = minval+tf.math.maximum(self.df['M33'], 0)
        self.df['M44'] = minval+tf.math.maximum(self.df['M44'], 0)

        self.df['sigma'+self.labels_list[0]] = abs(self.df['M11'])
        self.df['sigma'+self.labels_list[1]] = np.sqrt(self.df['M21']**2 + self.df['M22']**2)
        self.df['sigma'+self.labels_list[2]] = np.sqrt(self.df['M31']**2+self.df['M32']**2+self.df['M33']**2)
        self.df['sigma'+self.labels_list[3]] = np.sqrt(self.df['M41']**2+self.df['M42']**2+self.df['M43']**2+self.df['M44']**2)

        # calculates residuals for x, y, cotA, cotB
        self.residuals = np.empty(shape=(4, len(self.df["M11"])))
        self.residuals[0] = self.df[self.labels_list[0]+'true'] - self.df[self.labels_list[0]]
        self.residuals[1] = self.df[self.labels_list[1]+'true'] - self.df[self.labels_list[1]]
        self.residuals[2] = self.df[self.labels_list[2]+'true'] - self.df[self.labels_list[2]]
        self.residuals[3] = self.df[self.labels_list[3]+'true'] - self.df[self.labels_list[3]]

        mean0, std0 = (np.mean(np.abs(self.residuals[0])),np.std(np.abs(self.residuals[0])))
        print(f"Mean and standard deviation of residuals for {self.labels_list[0]}: ({mean0},{std0})")

        mean1, std1 = (np.mean(np.abs(self.residuals[1])),np.std(np.abs(self.residuals[1])))
        print(f"Mean and standard deviation of residuals for {self.labels_list[1]}: ({mean1},{std1})")

        mean2, std2 = (np.mean(np.abs(self.residuals[2])),np.std(np.abs(self.residuals[2])))
        print(f"Mean and standard deviation of residuals for {self.labels_list[2]}: ({mean2},{std2})")

        mean3, std3 = (np.mean(np.abs(self.residuals[3])),np.std(np.abs(self.residuals[3])))
        print(f"Mean and standard deviation of residuals for {self.labels_list[3]}: ({mean3},{std3})")
        

    def plotResiduals(self):
        #if self.residuals == None:
        #    self.checkResiduals()
        fig, ax = plt.subplots(nrows=2,ncols=4, figsize=(18,7))
        for i in range(4):
            sns.regplot(x=self.df[f'{self.labels_list[i]}true'], y=self.residuals[i], x_bins=np.linspace(-self.normalization[i],self.normalization[i],50), fit_reg=None, marker='.',color='b', ax=ax[0,i])
            ax[0,i].set_xlabel(f'True {self.labels_list[i]} {self.units_list[i]}')
            ax[0,i].set_ylabel(f'{self.labels_list[i]} residuals {self.units_list[i]}')

            ax[1,i].hist(self.residuals[i], bins = 25, align='mid', weights=1/len(self.residuals[i]) * np.ones(len(self.residuals[i])), histtype='step', color='b')
            ax[1,i].set_xlabel(f'{self.labels_list[i]} residuals {self.units_list[i]}')
            ax[1,i].set_ylabel("Fraction of clusters")
        fig.tight_layout(pad=2.0)
        plt.show()


class ClassificationModel(RegressionModel):
    def __init__(self, 
                 data_directory_path: str = "./", 
                 is_directory_recursive: bool = False, 
                 file_type: str = "parquet", 
                 data_format: str = "3D", # can't get 2D working
                 batch_size: int = 300, 
                 labels_list: list = None, 
                 units_list: list = None, 
                 normalization: int = 1, 
                 muon_collider: bool = True, 
                 file_fraction: float = .8, # fraction of files used for training
                 to_standardize: bool = False, 
                 input_shape: tuple = (1,13, 21), # dimension of 3D cluster
                 transpose=(0, 2, 3, 1), # not sure what this does 
                 use_time_stamps=[19], # last timestamp is 19
                 output_dir: str = "./filtering_models", 
                 learning_rate: float = 0.001, 
                 tag: str = "", 
                 x_feature_description: list = ['x_size', 'y_size', 'y_local', 'z_global'],
                 filteringBIB: bool = True,
                 training_dir: str = None,
                 model_number: int = None,
                 stamp: str = None,
                 cut_hit_time: bool = False,
                 hit_time_version: str = "adjusted_hit_time"
                 ):
        
        super().__init__(data_directory_path, 
                         is_directory_recursive, 
                         file_type, 
                         data_format, 
                         batch_size, 
                         labels_list, 
                         units_list,
                         normalization, 
                         muon_collider, 
                         file_fraction,
                         to_standardize, 
                         input_shape, 
                         transpose, 
                         use_time_stamps, 
                         output_dir, 
                         learning_rate, 
                         tag, 
                         x_feature_description,
                         filteringBIB,
                         training_dir,
                         model_number,
                         stamp,
                         cut_hit_time,
                         hit_time_version
                         )
    
    def createModel(self, layer1=3, layer2=3, numLayers=1):
        start_time = time.time()
        self.model=CreateClassificationModel(self.x_features, layer1, layer2, numLayers)
        self.model.summary()
        print("--- Model create and compile %s seconds ---" % (time.time() - start_time))

    def compileModel(self, learning_rate = 0.001):
        print(f"Compiling model with learning rate: {learning_rate}")
        self.learning_rate = learning_rate
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['binary_accuracy'])
    
    def loadWeights(self, weightsFile):
        self.model.load_weights(weightsFile)

    def saveWeights(self, fileName):
        filePath = Path(self.training_dir, f'weights_{fileName}.hdf5').resolve()
        self.model.save_weights(filePath)

    def runTraining(self, epochs=50, early_stopping=True, save_all_weights=False, schedule_lr=True, save_prev_history=False):
        super().runTraining(epochs=epochs, early_stopping=early_stopping, save_all_weights=save_all_weights, schedule_lr=schedule_lr, save_prev_history=save_prev_history)

    def checkAccuracy(self, threshold=0.5):
        p_test = self.model.predict(self.validation_generator)

        complete_truth = None
        for _, y in tqdm(self.validation_generator):
            if complete_truth is None:
                complete_truth = y
            else:
                complete_truth = np.concatenate((complete_truth, y), axis=0)

        prediction=p_test.flatten()
        labels = complete_truth.flatten()

        # background regection
        backgroundCount = 0
        rejectedBackground = 0
        # signal efficiency
        acceptedSignal = 0
        signalCount = 0

        for l, p in zip(labels, prediction):
            # background
            if l <= threshold:
                backgroundCount += 1
                if p <= threshold:
                    rejectedBackground += 1
            else:
                signalCount += 1
                if p > threshold:
                    acceptedSignal += 1

        signalEfficiency = acceptedSignal/signalCount*100
        backgroundRejection = rejectedBackground/backgroundCount*100

        accuracy = (acceptedSignal+rejectedBackground)/(signalCount+backgroundCount)*100

        fractionSignal = signalCount/(signalCount+backgroundCount)*100

        print(f"\nSignal Efficiency: {round(signalEfficiency,2)}%\nBackground Rejection: {round(backgroundRejection,2)}%\n")

        print(f"Overall Accuracy: {round(accuracy,2)}%\nFraction of Data that are Signal: {round(fractionSignal,2)}%")

        print(f"\nTotal number of clusters: {signalCount+backgroundCount}")


    def checkAccuracyTrainingData(self, threshold=0.5):
        p_test = self.model.predict(self.training_generator)

        complete_truth = None
        for _, y in tqdm(self.training_generator):
            if complete_truth is None:
                complete_truth = y
            else:
                complete_truth = np.concatenate((complete_truth, y), axis=0)

        prediction=p_test.flatten()
        labels = complete_truth.flatten()

        # background regection
        backgroundCount = 0
        rejectedBackground = 0
        # signal efficiency
        acceptedSignal = 0
        signalCount = 0

        for l, p in zip(labels, prediction):
            # background
            if l <= threshold:
                backgroundCount += 1
                if p <= threshold:
                    rejectedBackground += 1
            else:
                signalCount += 1
                if p > threshold:
                    acceptedSignal += 1

        signalEfficiency = acceptedSignal/signalCount*100
        backgroundRejection = rejectedBackground/backgroundCount*100

        accuracy = (acceptedSignal+rejectedBackground)/(signalCount+backgroundCount)*100

        fractionSignal = signalCount/(signalCount+backgroundCount)*100

        print(f"\nSignal Efficiency: {round(signalEfficiency,2)}%\nBackground Rejection: {round(backgroundRejection,2)}%\n")

        print(f"Overall Accuracy: {round(accuracy,2)}%\nFraction of Data that are Signal: {round(fractionSignal,2)}%")

        print(f"\nTotal number of clusters: {signalCount+backgroundCount}")


    def countClusters(self):
        complete_truth = None
        for _, y in tqdm(self.training_generator):
            if complete_truth is None:
                complete_truth = y
            else:
                complete_truth = np.concatenate((complete_truth, y), axis=0)

        labelst = complete_truth.flatten()

        complete_truth = None
        for _, y in tqdm(self.validation_generator):
            if complete_truth is None:
                complete_truth = y
            else:
                complete_truth = np.concatenate((complete_truth, y), axis=0)

        labelsv = complete_truth.flatten()
        
        print(f"# of training clusters: {len(labelst)}")
        print(f"# of training clusters: {len(labelsv)}")
    

    def plotTraining(self):
        fig, ax = plt.subplots(ncols=1, nrows=2, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax[0].plot(range(1,len(self.history.history['val_binary_accuracy'])+1),self.history.history['val_binary_accuracy'],c='royalblue')
        ax[0].set_ylabel("Validation Binary Accuracy")
        ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        
        ax[1].plot(range(1,len(self.history.history['val_binary_accuracy'])+1),self.history.history['learning_rate'],c='seagreen')
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Learning Rate")
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[1].yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0e}'.format(x)))

        ax[0].grid(True, linestyle='--', linewidth=0.5, color='gray')
        ax[1].grid(True, linestyle='--', linewidth=0.5, color='gray')

        fig.tight_layout()

        plt.show()

    
    def plotROCcurve(self):
        complete_truth = None
        for _, y in tqdm(self.validation_generator):
            if complete_truth is None:
                complete_truth = y
            else:
                complete_truth = np.concatenate((complete_truth, y), axis=0)
        labels = complete_truth.flatten()

        fig, ax = plt.subplots()

        p_test = self.model.predict(self.validation_generator)
            
        prediction=p_test.flatten()

        fpr_keras, tpr_keras, thresholds_keras = roc_curve(labels, prediction)
        auc_keras = auc(fpr_keras, tpr_keras)

        # get index of threshold=0.5
        selected_threshold = 0.5
        closest = thresholds_keras[min(range(len(thresholds_keras)), key = lambda i: abs(thresholds_keras[i]-selected_threshold))]
        selected_index = np.where(thresholds_keras == closest)[0]

        # get index of best threshold
        temp = tpr_keras-fpr_keras
        max_value = max(temp)
        best_index = np.where(temp == max_value)[0]

        print(f"Optimal threshold: {thresholds_keras[best_index][0]}")

        ax.plot(fpr_keras, tpr_keras, label=f'ROC curve (area = {round(auc_keras,3)})')
        ax.scatter(fpr_keras[selected_index], tpr_keras[selected_index], label='selected threshold: 0.5',c='orange')
        ax.scatter(fpr_keras[best_index], tpr_keras[best_index], label=f'optimal threshold: {round(thresholds_keras[best_index][0],2)}', c='green')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.minorticks_on()
        ax.grid(which='both', color='lightgray', linestyle='--', linewidth=0.5)
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_title('ROC curve')
        ax.legend(loc='best')
        plt.show()


def CompareModelROCCurves(models):
    fig, ax = plt.subplots()

    for model in models:
        name=model.modelName

        complete_truth = None
        for _, y in tqdm(model.validation_generator):
            if complete_truth is None:
                complete_truth = y
            else:
                complete_truth = np.concatenate((complete_truth, y), axis=0)
        labels = complete_truth.flatten()

        p_test = model.model.predict(model.validation_generator)
            
        prediction=p_test.flatten()

        fpr_keras, tpr_keras, thresholds_keras = roc_curve(labels, prediction)
        auc_keras = auc(fpr_keras, tpr_keras)

        # get index of best threshold
        temp = tpr_keras-fpr_keras
        max_value = max(temp)
        best_index = np.where(temp == max_value)[0]

        print(f"Optimal threshold for {name}: {thresholds_keras[best_index][0]}")

        ax.plot(fpr_keras, tpr_keras, label=f'{name} (area = {round(auc_keras,3)})')

    ax.plot([0, 1], [0, 1], 'k--')
    ax.minorticks_on()
    ax.grid(which='both', color='lightgray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curve')
    ax.legend(loc='best')
    plt.show()
