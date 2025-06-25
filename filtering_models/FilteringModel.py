import os

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import roc_curve, auc

from qkeras import *
import OptimizedDataGenerator4 as ODG
import time

# python based
from pathlib import Path
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
plt.rcParams['figure.dpi'] = 300
from tqdm import tqdm
import numpy as np

# custom code
from models import *
"""
def scheduler(epoch, lr):
 
    if epoch == 30 or epoch == 60 or epoch == 90:
        return lr/2
    else:
        return lr
"""  

# This just allows plotting the learning rate after training finishes
class LearningRate(tf.keras.callbacks.Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        lr = float(self.model.optimizer.learning_rate)
        logs['lr'] = lr
     
class FilteringModel:

    def __init__(self, 
                 learning_rate: float = 0.01, 
                 x_feature_description: list = ['x_size', 'y_size', 'y_local', 'z_global'],
                 tf_records_dir: str = None,
                 model_number: int = None,
                 y_local_layer: list = [3,1],
                 z_global_layer: list = [3,1],
                 x_profile_layer: list = [3,1],
                 y_profile_layer: list = [3,1],
                 classification_layer: list = [3,1],
                 pairInputs: bool = True,
                 addDirectPath: bool = False,
                 nSteps: int = 500,
                 clipnorm: float = None,
                 loadModel: bool = False,
                 verbose = 1
                 ):

        self.history = None
        self.verbose = verbose

        training_dir = Path(tf_records_dir, 'tfrecords_train')
        validation_dir = Path(tf_records_dir, 'tfrecords_validation')

        # get number of batches from number of tf records files
        self.nBatches=len([f for f in os.listdir(training_dir) if ".tfrecord" in f])
        print("batches: ", self.nBatches)

        # create the set of training data
        self.training_generator = ODG.OptimizedDataGenerator(load_records=True, tf_records_dir=training_dir, x_feature_description=x_feature_description)

        # Create the validation data set, with different features
        self.validation_generator = ODG.OptimizedDataGenerator(load_records=True, tf_records_dir=validation_dir, x_feature_description=x_feature_description)
        self.all_features=["x_profile", "y_profile", "x_size", "y_size", "y_local", "z_global", "total_charge", "adjusted_hit_time", "adjusted_hit_time_30ps_gaussian", "adjusted_hit_time_60ps_gaussian"]
        self.validation_generator_all_features = ODG.OptimizedDataGenerator(load_records=True, tf_records_dir=validation_dir, x_feature_description=self.all_features)
        
        # in the tf records the data is scaled to all range from 0 to 1, but we want to be able to scale it back if needed, so here are the factors to multiply by
        self.scaling={"x_size":21, "y_size":13, "y_local":8.5, "z_global":65, "total_charge":150000}
        
        self.x_features=x_feature_description

        self.nSteps = nSteps
        self.clipnorm = clipnorm

        if not loadModel:
            self.createModel(pairInputs, addDirectPath, y_local_layer,z_global_layer,x_profile_layer,y_profile_layer,classification_layer)
        else:
            self.loadModel(model_number)

        self.modelName=f"Model {model_number}" 

        self.compileModel(learning_rate=learning_rate)

        self.info = None

    # Testing library that does hyperparameter optimization on it's own
    def optimizeKerasTunerModel(self):
        stamp = os.urandom(3).hex()
        folder = f"/home/elizahoward/smart-pixels-ml/filtering_models/hyperparameter_optimization_{stamp}"
        os.makedirs(folder)
        tuner = kt.Hyperband(model_builder,
                     objective='val_binary_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory=folder,
                     project_name='kt_test') 
        
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(x=self.training_generator,validation_data=self.validation_generator, epochs=50, validation_split=0.2, callbacks=[stop_early])
        # Get the optimal hyperparameters
        self.best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

        print(f"""
        The hyperparameter search is complete. The optimal number of units in the first densely-connected
        layer is {self.best_hps.get('units')} and the optimal learning rate for the optimizer
        is {self.best_hps.get('learning_rate')}.
        """)
    
    # use models.py file to make the model
    def createModel(self, pairInputs,addDirectPath, y_local_layer,z_global_layer,x_profile_layer,y_profile_layer,classification_layer):
        self.model=CreateClassificationModel(self.x_features, pairInputs, addDirectPath,y_local_layer,z_global_layer,x_profile_layer,y_profile_layer,classification_layer)
        if self.verbose:
            self.model.summary()

    def compileModel(self, learning_rate = 0.01):
        if self.verbose:
            print(f"Compiling model with learning rate: {learning_rate}")
        self.learning_rate = learning_rate
        
        # the number of epochs you want to decay over = number of steps in cosine decay * the batch size
        decay_steps = self.nSteps*self.nBatches

        # we can have the learning rate start lower and "warm up" to the learning rate we want before it decays
        warmup_steps = int(self.nSteps*self.nBatches/10)
        warmup_target = learning_rate

        lr_scheduler = keras.optimizers.schedules.CosineDecay(initial_learning_rate=learning_rate/3, decay_steps=decay_steps,warmup_steps=warmup_steps, alpha=learning_rate/5, warmup_target=warmup_target)
        
        self.model.compile(optimizer=Adam(learning_rate=lr_scheduler,clipnorm=self.clipnorm), loss='binary_crossentropy', metrics=['binary_accuracy'])

    def loadWeights(self, weightsFile):
        self.model.load_weights(weightsFile)
        self.info=None 
        self.getInfo()

    def loadModel(self, model_number):
        self.modelName=f"Model {model_number}"
        file_path = Path(f'./Model_{model_number}.keras').resolve()
        self.model=tf.keras.models.load_model(file_path, compile=False)

    def saveModel(self, model_number, overwrite=False):
        file_path = Path(f'./Model_{model_number}.keras').resolve()
        if not overwrite and os.path.exists(file_path):
            raise Exception("Model exists. To overwrite existing saved model, set overwrite to True.")
        self.modelName=f"Model {model_number}"
        self.model.save(file_path)
    
    def saveWeights(self, folder=None, fileName = "", overwrite=False):
        if folder==None:
            file_path = Path(f'./{self.modelName}{fileName}.weights.h5').resolve()
        else:
            file_path = Path(f'{folder}/{self.modelName}{fileName}.weights.h5').resolve()
        if not overwrite and os.path.exists(file_path):
            raise Exception("Weights file exists. To overwrite existing saved model, set overwrite to True.")
        self.model.save_weights(file_path)

    def runTraining(self, epochs=None, early_stopping=True, save_learning_rate=True, save_weights_files=False,plotTraining=False):
        if epochs is None:
            epochs = int(self.nSteps*1.1)

        # we can stop training early if training is not improving the weights 
        early_stopping_patience = 20
        es = EarlyStopping( 
        patience=early_stopping_patience,
        restore_best_weights=True
        )
    
        # checkpoint path if saving weights at each epoch (generally, don't do this)
        checkpoint_filepath = Path("./weights", 'epoch.{epoch:02d}-t{loss:.2f}-v{val_loss:.2f}.weights.h5').resolve()
        mcp = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            save_best_only=True,
        )
            
        callbacks = []
        if early_stopping:
            callbacks.append(es)
        if save_learning_rate:
            callbacks.append(LearningRate())
        if save_weights_files:
            callbacks.append(mcp)

        self.history = self.model.fit(x=self.training_generator,
                        validation_data=self.validation_generator,
                        callbacks=callbacks,
                        epochs=epochs,
                        shuffle=False,
                        verbose=self.verbose)

        self.info=None
        if plotTraining:
            self.plotTraining() 
        

    def plotTraining(self, folder=None, savePlot=False):
        all_same = True

        if 'lr' in self.history.history.keys():
            lr_i = self.history.history['lr'][0]
            all_same = True
            for lr in self.history.history['lr']:
                if lr != lr_i:
                    all_same=False
                    break
        
        if all_same:
            fig, ax = plt.subplots()
            ax.plot(range(1,len(self.history.history['val_binary_accuracy'])+1),self.history.history['val_binary_accuracy'],c='royalblue',label='validation')
            ax.plot(range(1,len(self.history.history['binary_accuracy'])+1),self.history.history['binary_accuracy'],c='green',label='training')
            ax.set_ylabel("Binary Accuracy")
            ax.legend(loc='lower right')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("Epoch")
            ax.set_title(f"Learning Rate: {self.learning_rate}")
            ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

        else:
            fig, ax = plt.subplots(ncols=1, nrows=2, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
            ax[0].plot(range(1,len(self.history.history['val_binary_accuracy'])+1),self.history.history['val_binary_accuracy'],c='seagreen',label='validation')
            ax[0].plot(range(1,len(self.history.history['binary_accuracy'])+1),self.history.history['binary_accuracy'],c='royalblue',label='training')
            ax[0].set_ylabel("Binary Accuracy")
            ax[0].legend(loc='lower right')
            ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[0].set_title(f"{self.modelName}")

            
            ax[1].plot(range(1,len(self.history.history['val_binary_accuracy'])+1),self.history.history['lr'],c='violet')
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Learning Rate")
            ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[1].yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0e}'.format(x)))

            ax[0].grid(True, linestyle='--', linewidth=0.5, color='gray')
            ax[1].grid(True, linestyle='--', linewidth=0.5, color='gray')

            fig.tight_layout()
        
        if savePlot:
            if folder==None:
                fileName = f'./{self.modelName}_CosineDecay{self.nSteps}Steps_clipnorm{self.clipnorm}.png'
            else:
                fileName=f'{folder}/{self.modelName}_CosineDecay{self.nSteps}Steps_clipnorm{self.clipnorm}.png'
            plt.savefig(fileName, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    # This needs a better name, but I can't think of one 
    # This puts the validation clusters through the model, gets predictions, and also pulls other info
    def getInfo(self): 
        
        if self.info is None: 

            self.info = {}

            # get predictions (use regular validation generator)
            p_test = self.model.predict(self.validation_generator)
            prediction=p_test.flatten()

            # Iterate through feature info and label on each cluster (use validation generator with the extra info)
            for x_features, y in tqdm(self.validation_generator_all_features):
                #features_=[*tuple] 
                if not bool(self.info):
                    self.info['labels']=y 
                    for x_feature in x_features.keys():
                        self.info[x_feature]=x_features[x_feature]
                else:
                    self.info['labels'] = np.concatenate((self.info['labels'], y), axis=0) 
                    for x_feature in x_features.keys():
                        self.info[x_feature]=np.concatenate((self.info[x_feature], x_features[x_feature]), axis=0)
            for item in list(self.info.keys()):
                if self.info[item].shape[1]==1:
                    self.info[item] = np.array(self.info[item]).flatten()

        
            for item in list(self.scaling.keys()):
                self.info[item]*=self.scaling[item]

            self.info['prediction']=prediction

    # Check model accuracy on validation data, optionally taking into account a cut on the hit time
    def checkAccuracy(self, threshold=0.5): 
        self.getInfo()
        
        rejectedBackgrounds = np.array([0,0,0])
        acceptedSignals = np.array([0,0,0])

        bibClusters={}
        bibClusters['prediction']=self.info['prediction'][self.info['labels']==0]
        bibClusters['adjusted_hit_time_30ps_gaussian']=self.info['adjusted_hit_time_30ps_gaussian'][self.info['labels']==0]
        totalBackground = len(bibClusters['prediction'])
        rejectedBackgrounds[0] = len(bibClusters['prediction'][bibClusters['prediction']<threshold])
        rejectedBackgrounds[1] = len(bibClusters['prediction'][(bibClusters['adjusted_hit_time_30ps_gaussian']<-3*30e-3) | (bibClusters['adjusted_hit_time_30ps_gaussian']>3*30e-3)])
        rejectedBackgrounds[2] = len(bibClusters['prediction'][(bibClusters['adjusted_hit_time_30ps_gaussian']<-3*30e-3) | (bibClusters['adjusted_hit_time_30ps_gaussian']>3*30e-3) | (bibClusters['prediction']<threshold)])
        
        sigClusters = {}
        sigClusters['prediction']=self.info['prediction'][self.info['labels']==1]
        sigClusters['adjusted_hit_time_30ps_gaussian']=self.info['adjusted_hit_time_30ps_gaussian'][self.info['labels']==1]
        totalSignal = len(sigClusters['prediction'])
        acceptedSignals[0] = len(sigClusters['prediction'][sigClusters['prediction']>=threshold])
        acceptedSignals[1] = len(sigClusters['prediction'][(sigClusters['adjusted_hit_time_30ps_gaussian']>-3*30e-3) & (sigClusters['adjusted_hit_time_30ps_gaussian']<3*30e-3)])
        acceptedSignals[2] = len(sigClusters['prediction'][(sigClusters['adjusted_hit_time_30ps_gaussian']>-3*30e-3) & (sigClusters['adjusted_hit_time_30ps_gaussian']<3*30e-3) & (sigClusters['prediction']>=threshold)])
        

        signalEfficiency = acceptedSignals/totalSignal*100
        backgroundRejection = rejectedBackgrounds/totalBackground*100

        accuracy = (acceptedSignals+rejectedBackgrounds)/(totalSignal+totalBackground)*100  

        fractionSignal = totalSignal/(totalSignal+totalBackground)*100

        # Print fraction of signal to see if the network is just predicting everything is signal/bib
        print(f"Overall Accuracy of Neural Network: {round(accuracy[0],2)}%\nFraction of Data that are Signal: {round(fractionSignal,2)}%\n")

        print(f"Neural Network results without cutting based on hit time: \n")

        print(f"\nSignal Efficiency: {round(signalEfficiency[0],2)}%\nBackground Rejection: {round(backgroundRejection[0],2)}%\n\n")

        print(f"Neural Network results using hit time with 30ps gaussian: \n")

        print(f"\nSignal Efficiency: {round(signalEfficiency[1],2)}%\nBackground Rejection: {round(backgroundRejection[1],2)}%\n\n")

        print(f"Cut results by themselves for 30 ps Gaussian: \n")

        print(f"\nSignal Efficiency: {round(signalEfficiency[2],2)}%\nBackground Rejection: {round(backgroundRejection[2],2)}%\n\n")

        #print(f"Total number of clusters: {signalCount+backgroundCount}")


    # Sorts BIB/signal clusters by whether they were accepted or rejected 
    def compareAcceptedRejectedClusters(self, threshold=0.5):
        self.getInfo()

        # Make dict of just bib clusters
        self.bibInfo={} 
        cut=self.info['labels']==0
        for item in list(self.info.keys()):
            self.bibInfo[item]=self.info[item][cut]
        
        # Make dict of just signal clusters
        self.sigInfo={} 
        cut=self.info['labels']==1
        for item in list(self.info.keys()):
            self.sigInfo[item]=self.info[item][cut]
        
        # Sort by whether they were accepted or rejected
        self.rejected_by_nn={}
        self.rejected_by_time={}
        self.rejected_by_both={}
        self.not_rejected={}
        cut1=self.bibInfo['prediction']<threshold 
        cut2=np.abs(self.bibInfo['adjusted_hit_time_30ps_gaussian'])>90e-3
        for item in list(self.info.keys()):
            self.rejected_by_nn[item]=self.bibInfo[item][cut1]
            self.rejected_by_time[item]=self.bibInfo[item][cut2]
            self.rejected_by_both[item]=self.bibInfo[item][cut1 | cut2]
            self.not_rejected[item]=self.bibInfo[item][~(cut1 | cut2)]


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


    def plotROCcurve(self,weightsFiles=None,folder=None,threshold=0.5):
        if weightsFiles is None:
            
            self.getInfo()
            fpr_keras, tpr_keras, thresholds_keras = roc_curve(self.info['labels'], self.info['prediction'])
            auc_keras = auc(fpr_keras, tpr_keras)

            # get index of threshold
            selected_threshold = threshold
            closest = thresholds_keras[min(range(len(thresholds_keras)), key = lambda i: abs(thresholds_keras[i]-selected_threshold))]
            selected_index = np.where(thresholds_keras == closest)[0]

            # get index of best threshold
            temp = tpr_keras-fpr_keras
            max_value = max(temp)
            best_index = np.where(temp == max_value)[0]
            print(f"Optimal threshold: {thresholds_keras[best_index][0]}")

            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1], 'k--')
            ax.plot(fpr_keras, tpr_keras, label=f'ROC curve (area = {round(auc_keras,3)})')
            ax.scatter(fpr_keras[selected_index], tpr_keras[selected_index], label='selected threshold: 0.5',c='orange')
            ax.scatter(fpr_keras[best_index], tpr_keras[best_index], label=f'optimal threshold: {round(thresholds_keras[best_index][0],2)}', c='green')
            ax.minorticks_on()
            ax.grid(which='both', color='lightgray', linestyle='--', linewidth=0.5)
            ax.set_xlabel('False positive rate')
            ax.set_ylabel('True positive rate')
            ax.set_title('ROC curve')
            ax.legend(loc='best')
            plt.show()

        else:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1], 'k--')

            stepsClipnormArray = []
            aucArray = []
            
            for weightsFile in weightsFiles:
                self.loadWeights(weightsFile)
                steps = weightsFile[:weightsFile.find('steps')]
                steps = steps[steps.find('Model ')+1:]
                steps = int(steps[steps.find('_')+1:])
                clipnorm = float(weightsFile[weightsFile.find('clipnorm')+8:weightsFile.find('.weights')])
                print(steps,clipnorm)
                stepsClipnormArray.append((steps,clipnorm))
                fpr_keras, tpr_keras, thresholds_keras = roc_curve(self.info['labels'], self.info['prediction'])
                auc_keras = auc(fpr_keras, tpr_keras)
                aucArray.append(auc_keras)
                ax.plot(fpr_keras, tpr_keras, label=f'{steps} steps, clipnorm: {round(clipnorm,2)} (area = {round(auc_keras,3)})')

            ax.minorticks_on()
            ax.grid(which='both', color='lightgray', linestyle='--', linewidth=0.5)
            ax.set_xlabel('False positive rate')
            ax.set_ylabel('True positive rate')
            ax.set_title('ROC curve')
            ax.legend(loc='best')
            plt.savefig(f"{folder}/{self.modelName} ROC Curve.png")

            bestAUC = max(aucArray)
            bestIndex = aucArray.index(bestAUC)
            bestStepsClipnorm = stepsClipnormArray[bestIndex]
            return bestStepsClipnorm, auc_keras               

    def getBestThresholdWithHitTimeCut(self):
        self.getInfo()
        thresholds = np.arange(0, 1.05, 0.05)
        signalEfficiencys=[]
        backgroundRejections=[]
        # background regection
        backgroundCount = 0
        rejectedBackgrounds = np.zeros(len(thresholds))
        # signal efficiency
        signalCount = 0
        acceptedSignals = np.zeros(len(thresholds))
        
        for l, p, t in zip(self.info['labels'], self.info['prediction'], self.info["adjusted_hit_time_30ps_gaussian"]):
            # background
            if l <= 0.5:
                backgroundCount += 1
                for i in range(len(thresholds)):
                    threshold = thresholds[i]
                    if t<-3*30e-3 or t>3*30e-3 or p <= threshold:
                        rejectedBackgrounds[i] += 1
            else:
                signalCount += 1
                for i in range(len(thresholds)):
                    threshold = thresholds[i]
                    if t>-3*30e-3 and t<3*30e-3 and p > threshold:
                        acceptedSignals[i] += 1
        print(signalCount,acceptedSignals)
        signalEfficiencys=acceptedSignals/signalCount*100
        backgroundRejections=rejectedBackgrounds/backgroundCount*100
        temp = signalEfficiencys-backgroundRejections
        max_value = max(temp)
        best_index = np.where(temp == max_value)[0]
        print(f"Optimal threshold: {thresholds[best_index]}")
        print(f"Signal Efficiency: {signalEfficiencys[best_index]}%\nBackground Rejection: {backgroundRejections[best_index]}%\n")

    def plotFeatures(self):
        self.compareRejectedBIB()
        
        fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'hspace': 0.05, 'wspace': 0.03}, figsize=(10,7))

        ax[0,0].hist2d(self.not_rejected['z_global'], self.not_rejected['x_size'], bins=[30, np.arange(0,9,1)], cmap='Blues')
        ax[0,0].set_ylabel('x-size (# pixels)', fontsize=15)
        ax[0,0].set_title("BIB that was not rejected", fontsize=15)

        ax[0,1].hist2d(self.sigInfo['z_global'], self.sigInfo['x_size'], bins=[30, np.arange(0,9,1)], cmap='Blues')
        ax[0,1].set_title("Signal", fontsize=15)

        ax[1,0].hist2d(self.not_rejected['z_global'], self.not_rejected['y_size'], bins=[30,np.arange(0,14,1)], cmap='Blues')
        ax[1,0].set_xlabel('z-global [mm]', fontsize=15)
        ax[1,0].set_ylabel('y-size (# pixels)', fontsize=15)

        ax[1,1].hist2d(self.sigInfo['z_global'], self.sigInfo['y_size'], bins=[30,np.arange(0,14,1)], cmap='Blues')
        ax[1,1].set_xlabel('z-global [mm]', fontsize=15)

        plt.show()

        fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'hspace': 0.05, 'wspace': 0.03}, figsize=(10,7))

        ax[0,0].hist2d(self.not_rejected['y_local'], self.not_rejected['x_size'], bins=[30, np.arange(0,9,1)], cmap='Blues')
        ax[0,0].set_ylabel('x-size (# pixels)', fontsize=15)
        ax[0,0].set_title("BIB that was not rejected", fontsize=15)

        ax[0,1].hist2d(self.sigInfo['y_local'], self.sigInfo['x_size'], bins=[30, np.arange(0,9,1)], cmap='Blues')
        ax[0,1].set_title("Signal", fontsize=15)

        ax[1,0].hist2d(self.not_rejected['y_local'], self.not_rejected['y_size'], bins=[30,np.arange(0,14,1)], cmap='Blues')
        ax[1,0].set_xlabel('y-local [\u03bcm]', fontsize=15)
        ax[1,0].set_ylabel('y-size (# pixels)', fontsize=15)

        ax[1,1].hist2d(self.sigInfo['y_local'], self.sigInfo['y_size'], bins=[30,np.arange(0,14,1)], cmap='Blues')
        ax[1,1].set_xlabel('y-local [\u03bcm]', fontsize=15)

        plt.show()


def CompareModelROCCurves(models, fileName=None):
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
    if fileName is not None:
        plt.savefig(fileName)
    else:
        plt.show()
