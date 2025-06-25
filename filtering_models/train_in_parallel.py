"""
This is not currently working, needs debugging
"""

import sys
import psutil
import multiprocessing
import time 
import numpy as np

overwrite=False
if len(sys.argv) > 1:
    overwrite = sys.argv[1] == "overwrite" # If you want to overwrite a previous folder, use this
    stamp = sys.argv[1] == "stamp" # If you want to just generate a random stamp for the folder name

sys.path.insert(0, '/home/elizahoward/smart-pixels-ml/filtering_models') 
sys.path.insert(0, '/home/elizahoward/smart-pixels-ml') 
from FilteringModel import *
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import multiprocessing

import os, shutil

# Print the PID incase something goes wrong and you need to quit the program
pid = os.getpid()
print(f"PID: {pid}\n")

# Print number of cpu cores on the computer (should be 64)
cpu_cores = multiprocessing.cpu_count()
print(f"{cpu_cores} CPU cores available\n")

if stamp:
    stamp = os.urandom(3).hex()
    folder = f"/home/elizahoward/smart-pixels-ml/filtering_models/parallel_training_{stamp}"
else:
    # Change the folder name here if necessary
    folder="/home/elizahoward/smart-pixels-ml/filtering_models/parallel_training_2" 

if not os.path.isdir(folder):
    print(f"Creating folder {folder} for training\n")
    os.makedirs(folder)
    os.makedirs(f"{folder}/ROC_curves")
    os.makedirs(f"{folder}/weights")
    os.makedirs(f"{folder}/training_plots")

elif overwrite:
    print(f"Overwriting folder {folder} for training\n")

    # Clear files from folder you are overwriting
    files = os.listdir(folder)
    for f in files:
        if os.path.isfile(f"{folder}/{f}"):
            os.remove(f"{folder}/{f}")
            files = os.listdir(folder)
        elif os.path.isdir(f"{folder}/{f}"):
            more_files = os.listdir(f"{folder}/{f}")
            for mf in more_files:
                if os.path.isfile(f"{folder}/{f}/{mf}"):
                    os.remove(f"{folder}/{f}/{mf}")

else:
    raise Exception(f"Folder {folder} already exists. Use 'overwrite' argument to overwrite the folder.") 

# Dictionary for the best models for each configuration (usefull for when trying multiple different training methods)
bestTrainedModels = {}

x_feature_description = ['x_size', 'y_size', 'y_local', 'z_global']

# Do you want to set number of layers equal to number of neurons at each layer?
layerNeuronEqual = False 

# Do you want each section (y_local/y_size, z_global/x_size, classification) to all have the same configuration?
allSame = True

# Do you want to specify the last layer seperatley? If so, specify configuration for classification layer
lastLayerConstant = False 
classification_layer = [5,1] # [number of neurons, number of layers]

# Do you want to vary the number of layers? If so, specify min and max
varyLayers = False
minLayers=1
maxLayers=1 # Defaut if not varying

# Do you want to vary the number of neurons at each layer? If so, specify min and max
varyNeurons = False
minNeurons=3
maxNeurons=3 # Default if not varying

# Create array of all neuron/layer configurations, depeding on which variables are being varied
if varyLayers and varyNeurons:
    if layerNeuronEqual:
        neuronsLayers = [[l,l] for l in range(minLayers,maxLayers+1)]
    else:
        neuronsLayers = [[n,l] for n in range(minNeurons,maxNeurons+1) for l in range(minLayers,maxLayers+1)]
elif varyLayers:
    neuronsLayers = [[maxNeurons,l] for l in range(minLayers,maxLayers+1)]
elif varyNeurons:
    neuronsLayers = [[n,maxLayers] for n in range(minNeurons,maxNeurons+1)]
else:
    neuronsLayers = [[maxNeurons,maxLayers]]


# Create all possible model configurations for all three layers together (y_local/y_size, z_global/x_size, classification)
if allSame and lastLayerConstant:
    modelConfigurations=[[nL, nL, classification_layer] for nL in neuronsLayers]
elif allSame:
    modelConfigurations=[[nL, nL, nL] for nL in neuronsLayers]
elif lastLayerConstant:
    modelConfigurations=[[nL1, nL2, classification_layer] for nL1 in neuronsLayers for nL2 in neuronsLayers]
else:
    modelConfigurations=[[nL1, nL2, nL3] for nL1 in neuronsLayers for nL2 in neuronsLayers for nL3 in neuronsLayers]

# Assign model number to each model configuration
modelNumbers=np.arange(1,len(modelConfigurations)+1)
modelConfigurationsWithModelNumbers = [[modelConfiguration,modelNum] for modelConfiguration, modelNum in zip(modelConfigurations, modelNumbers)]

# Document the model configurations
file = open(f"{folder}/Model_Description.txt", "w")
for modelConfiguration, modelNum in modelConfigurationsWithModelNumbers:
    file.write(f'Model {modelNum}:\n\ty_local_layer: {modelConfiguration[0]}\n\tz_global_layer: {modelConfiguration[1]}\n\tclassification_layer: {modelConfiguration[2]}\n\n')
file.close()

# Do you want to vary the number of steps in cosine decay? If so, specify min and max
varySteps = False
minSteps=100
maxSteps=100 # Default if not varying
stepsStep=10

# Make array of all possible cosine decay steps
if varySteps:
    steps=np.arange(minSteps, maxSteps+stepsStep, stepsStep)
else:
    steps=[maxSteps]

# Do you want to vary clipnorm? If so, specify min and max
varyClipnorm = False
minClipnorm=0.5
maxClipnorm=0.5 # Default if not varying
clipnormStep=0.1

# Make array of all possible values of clipnorm
if varyClipnorm:
    clipnorms = np.arange(minClipnorm, maxClipnorm+clipnormStep, clipnormStep)
else:   
    clipnorms = [maxClipnorm]

# Do you want to vary the learning rate? If so, specify min and max
varyLearningRate = True
minLR=0.005
maxLR=0.005 # Default if not varying
LRstep=0.005

# Make array of all possible values of the learning rate
if varyLearningRate:
    learning_rates = np.arange(minLR, maxLR+LRstep, LRstep)
else:
    learning_rates = [maxLR]

# Make array describing all possible model configurations (with corresponding model number) and training configurations
trainingConfigurations = [[modelConfigAndNumber,nSteps,clipnorm,learning_rate] for modelConfigAndNumber in modelConfigurationsWithModelNumbers for nSteps in steps for clipnorm in clipnorms for learning_rate in learning_rates]

# Temporary check on total number of model/trianing configurations 
print(f"Total number of training runs: {len(trainingConfigurations)}")
answer = None
while answer not in ["yes", "no"]:
    answer = input("Do you want to continue and start the trianing? (yes/no) \t")
if answer == "no": exit()

# For a specific training and model configuration, create and train model
def createModelAndTrain(trainingConfiguration):
    # Extract info
    modelConfigAndNumber,nSteps,clipnorm,learning_rate = trainingConfiguration
    modelConfiguration, modelNum = modelConfigAndNumber
    y_local_layer,z_global_layer,classification_layer = modelConfiguration

    # Create and train model
    print(f"Starting training for Model {modelNum} with {nSteps} steps, clipnorm: {clipnorm}, and learning_rate: {learning_rate} \n")
    model=FilteringModel(tf_records_dir=Path('/home/elizahoward/smart-pixels-ml/filtering_models/filtering_records2000').resolve(), x_feature_description=x_feature_description,learning_rate=learning_rate,model_number=modelNum,pairInputs=True,y_local_layer=y_local_layer,z_global_layer=z_global_layer,classification_layer=classification_layer,nSteps=nSteps,clipnorm=clipnorm,verbose=1)
    model.runTraining()
    model.plotTraining(savePlot=True,folder=f"{folder}/training_plots")

    fileName = ""
    if varyClipnorm:

        fileName += f'_clipnorm{str(round(clipnorm,2)).replace(".", "p")}'
    if varyLearningRate:
        fileName += f'_LR{str(learning_rate).replace(".", "p")}'
    if varySteps:
        fileName += f"_{nSteps}steps"
    
    print(fileName)
        
    model.saveWeights(folder=f"{folder}/weights", fileName=fileName)
    print(f"Training complete for Model {modelNum} with {nSteps} steps, a clipnorm of {clipnorm}, and a learning rate of {learning_rate}\n")


def createROCCurvesPerModel(modelConfigurationWithModelNumber):
    modelConfiguration,modelNum = modelConfigurationWithModelNumber
    y_local_layer,z_global_layer,classification_layer = modelConfiguration
    print(f"Creating ROC Curve for Model {modelNum}\n")
    model=FilteringModel(tf_records_dir=Path('/home/elizahoward/smart-pixels-ml/filtering_models/filtering_records2000').resolve(), x_feature_description=x_feature_description,model_number=modelNum,pairInputs=True,y_local_layer=y_local_layer,z_global_layer=z_global_layer,classification_layer=classification_layer,verbose=1)
    
    fileNames = [f"{folder}/weights/{filename}" for filename in os.listdir(f"{folder}/weights") if f"Model {modelNum}_" in filename]
    fileNames = sorted(fileNames, key=lambda x: int(''.join(filter(str.isdigit, x))))
    print("files:",fileNames)
    bestStepClipnorm, auc = model.plotROCcurve(weightsFiles=fileNames, folder=f"{folder}/ROC_curves")
    #if auc >.8:
    bestTrainedModels[modelNum] = bestStepClipnorm

    print(f"Finished ROC Curve for Model {modelNum}\n")


def createROCCurvesForBestModels():
    models = []
    for modelConfiguration,modelNum in modelConfigurationsWithModelNumbers:
        y_local_layer,z_global_layer,classification_layer = modelConfiguration
        if modelNum not in bestTrainedModels:
            continue
        nSteps, clipnorm = bestTrainedModels[modelNum]
        model=FilteringModel(tf_records_dir=Path('/home/elizahoward/smart-pixels-ml/filtering_models/filtering_records2000').resolve(), x_feature_description=x_feature_description,learning_rate=learning_rate,model_number=modelNum,pairInputs=True,y_local_layer=y_local_layer,z_global_layer=z_global_layer,classification_layer=classification_layer,nSteps=nSteps,verbose=0)
        weightsFiles = [f"{folder}/weights/{filename}" for filename in os.listdir(f"{folder}/weights") if f"Model {modelNum}_{nSteps}_clipnorm{clipnorm}" in filename and '.h5' in filename]
        model.loadWeights(weightsFiles[0])
        models.append(model)
    CompareModelROCCurves(models, f"{folder}/Best_Model_ROC_Curves.png")
    print(f"Finished ROC Curves for Best Models\n")


# Consider system load and available memory
def get_adaptive_pool_size():
    
    system_load = psutil.cpu_percent()

    if system_load < 50:
        return cpu_cores
    elif system_load < 75:
        return cpu_cores // 2
    else:
        return cpu_cores // 4

pool_size = get_adaptive_pool_size()
print(f"Creating multiprocessing.Pool with pool size: {pool_size} for ML training")

pool = multiprocessing.Pool(processes=pool_size)
pool.map_async(createModelAndTrain, trainingConfigurations)
pool.close()
pool.join()

print("ML training complete\n")

pool_size = get_adaptive_pool_size()
print(f"Creating multiprocessing.Pool with pool size: {pool_size} for making ROC curves")

pool = multiprocessing.Pool(processes=pool_size)
pool.map_async(createROCCurvesPerModel, modelConfigurationsWithModelNumbers)
pool.close()
pool.join()

print("ROC curves complete\n")

createROCCurvesForBestModels()

files = os.listdir(f'{folder}/weights')
files = [f for f in files if '.h5' in f]
if len(files) == 0:
    print(f"\n\nMODEL TRAINING FAILED\nDELETING FOLDER {folder}.")
    shutil.rmtree(folder)