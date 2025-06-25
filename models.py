# fix for keras v3.0 update
import os
#os.environ['TF_USE_LEGACY_KERAS'] = '1' 

import tensorflow as tf
import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import Sequence
from qkeras import *
import keras_tuner as kt

# from tensorflow.keras import datasets, layers, models

def var_network(var, hidden=10, output=2):
    var = Flatten()(var)
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(8, 0, 1)")(var)
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(8, 0, 1)")(var)
    return QDense(
        output,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
    )(var)

def conv_network(var, n_filters=5, kernel_size=3):
    var = QSeparableConv2D(
        n_filters,kernel_size,
        depthwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
        pointwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        depthwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        pointwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)
    var = QConv2D(
        n_filters,1,
        kernel_quantizer=quantized_bits(4, 0, alpha=1),
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)    
    return var

def CreatePredictionModel(shape, n_filters, pool_size, include_y_local):
    x_base = x_in = Input(shape)
    stack = conv_network(x_base, n_filters)
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
    if include_y_local:
        stack = Flatten()(stack)
        y_local_in = Input(shape=(1,), name="y_local_input")
        stack = Concatenate()([stack, y_local_in])
    stack = var_network(stack, hidden=16, output=14)
    model = Model(inputs=x_in, outputs=stack)
    return model

def CreateClassificationModel(input_features, pairInputs = True, addDirectPath=False,y_local_layer=[3,1], z_global_layer=[3,1], x_profile_layer=[3,1], y_profile_layer=[3,1], classification_layer=[3,1]):
    x_size = Input(shape=(1,), name="x_size")
    y_size = Input(shape=(1,), name="y_size")

    x_profile = Input(shape=(21,1), name="x_profile")
    y_profile = Input(shape=(13,1), name="y_profile")

    y_local = Input(shape=(1,), name="y_local")
    z_global = Input(shape=(1,), name="z_global")

    total_charge = Input(shape=(1,), name="total_charge")

    cluster = Input(shape=(13,21,1), name="cluster")

    inputs=[]
    stacks = []

    if 'x_size' in input_features:
        inputs.append(x_size)
    if 'y_size' in input_features:
        inputs.append(y_size)
    if 'x_profile' in input_features:
        inputs.append(x_profile)
    if 'y_profile' in input_features:
        inputs.append(y_profile)    
    if 'y_local' in input_features:
        inputs.append(y_local)
    if 'z_global' in input_features:
        inputs.append(z_global)
    if 'total_charge' in input_features:
        inputs.append(total_charge)
    if 'cluster' in input_features:
        inputs.append(cluster)

    if pairInputs and 'y_size' in input_features and 'y_local' in input_features:

        stack1 = Concatenate()([y_size, y_local])

        for num in range(y_local_layer[1]):
            stack1 = Dense(y_local_layer[0], activation='relu', kernel_initializer='glorot_uniform')(stack1)

        stacks.append(stack1)
        
        if addDirectPath:
            stacks.append(y_size)

    if pairInputs and 'x_size' in input_features and 'z_global' in input_features:
        stack2 = Concatenate()([x_size, z_global])

        for num in range(z_global_layer[1]):
            stack2 = Dense(z_global_layer[0], activation='relu', kernel_initializer='glorot_uniform')(stack2)
        
        stacks.append(stack2)
        
        if addDirectPath:
            stacks.append(x_size) 

    if pairInputs and 'y_profile' in input_features and 'y_local' in input_features:
        stack3 = Conv1D(filters=5, kernel_size=3, activation='relu')(y_profile)

        stack3 = MaxPooling1D(pool_size=2)(stack3)

        stack3 = Flatten()(stack3)
        
        """for num in range(y_profile_layer[1]):
            stack3 = Dense(y_profile_layer[0], activation='relu')(stack3)"""

        stack3 = Concatenate()([stack3, y_local])
        for num in range(y_local_layer[1]):
            stack3 = Dense(y_local_layer[0], activation='relu', kernel_initializer='glorot_uniform')(stack3)

        stacks.append(stack3)

    if pairInputs and 'x_profile' in input_features and 'z_global' in input_features:
        stack4 = Conv1D(filters=5, kernel_size=3, activation='relu')(x_profile)

        #stack4 = MaxPooling1D(pool_size=2)(stack4)

        stack4 = Flatten()(stack4)

        """for num in range(x_profile_layer[1]):
            stack4 = Dense(x_profile_layer[0], activation='relu')(stack4)"""

        stack4 = Concatenate()([stack4, z_global])

        for num in range(z_global_layer[1]):
            stack4 = Dense(z_global_layer[0], activation='relu', kernel_initializer='glorot_uniform')(stack4)

        stacks.append(stack4)

    if not pairInputs:
        stacks = inputs
        
    if 'total_charge' in input_features:
        stacks.append(total_charge)

    if 'cluster' in input_features and 'y_local' in input_features:

        stack5 = Conv2D(filters=5, kernel_size=3, activation='relu')(cluster)

        #stack5 = MaxPooling2D(pool_size=2)(stack5)

        stack5 = Flatten()(stack5)

        stack5 = Dense(5, activation='relu')(stack5)

        stack5 = Concatenate()([stack5, y_local])

        stack5 = Dense(3, activation='relu', kernel_initializer='glorot_uniform')(stack5)

        stacks.append(stack5)

    #stacks.append(x_size) # Test to see if this helps with the model

    stack = Concatenate()(stacks)
    
    for num in range(classification_layer[1]):
        stack = Dense(classification_layer[0], activation='relu', kernel_initializer='glorot_uniform')(stack)

    output = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')(stack)

    model = Model(inputs=inputs, outputs=output)

    return model

def model_builder(hp):

    x_size = Input(shape=(1,), name="x_size")
    y_size = Input(shape=(1,), name="y_size")

    y_local = Input(shape=(1,), name="y_local")
    z_global = Input(shape=(1,), name="z_global")

    inputs=[x_size, y_size, y_local, z_global]
    stacks = [x_size, y_size, y_local, z_global]
        
    stack = Concatenate()(stacks)

    hp_units = hp.Int('units', min_value=1, max_value=10, step=1)
    
    # num in range(3):
    stack = Dense(units=hp_units, activation='relu', kernel_initializer='glorot_uniform')(stack)

    output = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')(stack)

    model = Model(inputs=inputs, outputs=output)    

    hp_learning_rate = hp.Choice('learning_rate', values=[1, 1e-1, 1e-2])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='binary_crossentropy', metrics=['binary_accuracy'])

    return model