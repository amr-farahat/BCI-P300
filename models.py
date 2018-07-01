#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 13:57:10 2018

@author: amr
"""
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, GRU, Input
from keras.layers import Conv1D, Conv2D, MaxPooling2D, AveragePooling2D, AveragePooling1D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Reshape, Merge, SeparableConv2D
from keras.layers import PReLU
from keras.layers.merge import average, concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.local import LocallyConnected1D
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import Callback, TensorBoard, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.constraints import maxnorm
from keras.utils import plot_model
from keras import regularizers
from keras.regularizers import l2, l1
import keras.backend as K

def branched2(data_shape, model_config={'bn':True, 'dropout':True, 'branched':True, 'deep':True, 'nonlinear':'tanh'}, f=1):
    timepoints = data_shape[1]
    channels = data_shape[2]
    reg = 0.01
    input_data = Input(shape=(timepoints, channels, 1))


    spatial_conv = Conv2D(12, (1,channels),  padding='valid', kernel_regularizer=l2(reg))(input_data)
    if model_config['bn']:
        spatial_conv = BatchNormalization()(spatial_conv)
    spatial_conv = Activation(model_config['nonlinear'])(spatial_conv)
    if model_config['dropout']:
        spatial_conv = Dropout(0.5)(spatial_conv)

    if model_config['branched']:
        branch1 = Conv2D(4, (21*f,1), padding='valid', kernel_regularizer=l2(reg))(spatial_conv)
        if model_config['bn']:
            branch1 = BatchNormalization()(branch1)
        branch1 = Activation(model_config['nonlinear'])(branch1)
        branch1 = MaxPooling2D(pool_size=(3,1), strides=(3,1))(branch1)
        if model_config['dropout']:
            branch1 = Dropout(0.5)(branch1)


        branch1 = Flatten()(branch1)

    branch2 = Conv2D(4, (5*f,1), padding='valid', dilation_rate=(1,1), kernel_regularizer=l2(reg))(spatial_conv)
    if model_config['bn']:
        branch2 = BatchNormalization()(branch2)
    branch2 = Activation(model_config['nonlinear'])(branch2)
    branch2 = MaxPooling2D(pool_size=(3,1), strides=(3,1))(branch2)
    if model_config['dropout']:
        branch2 = Dropout(0.5)(branch2)

    if model_config['deep']:
        branch2 = Conv2D(8, (5*f,1), padding='valid', dilation_rate=(1,1), kernel_regularizer=l2(reg))(branch2)
        if model_config['bn']:
            branch2 = BatchNormalization()(branch2)
        branch2 = Activation(model_config['nonlinear'])(branch2)
        if model_config['dropout']:
            branch2 = Dropout(0.5)(branch2)
    #
        branch2 = Conv2D(8, (5*f,1), padding='valid', kernel_regularizer=l2(reg))(branch2)
        if model_config['bn']:
            branch2 = BatchNormalization()(branch2)
        branch2 = Activation(model_config['nonlinear'])(branch2)
        branch2 = MaxPooling2D(pool_size=(2,1), strides=(2,1))(branch2)
        if model_config['dropout']:
            branch2 = Dropout(0.5)(branch2)
#
    branch2 = Flatten()(branch2)

    if model_config['branched']:
        merged = concatenate([branch1,  branch2])
        dense = Dense(1, activation='sigmoid', kernel_regularizer=l2(reg))(merged)
    else:
        dense = Dense(1, activation='sigmoid', kernel_regularizer=l2(reg))(branch2)

    model = Model(inputs = [input_data], outputs=[dense])


    return model



def create_cnn(data_shape, f=1):

    timepoints = data_shape[1]
    channels = data_shape[2]
    kernel_size = 3*f
    model = Sequential()

    model.add(Conv2D(4, (kernel_size,3), activation='tanh', input_shape=(timepoints, channels, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))


    model.add(Conv2D(8, (kernel_size,3), strides=(1,1), activation='tanh', padding='valid'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
#    model.add(Dropout(0.25))

    model.add(Conv2D(16, (kernel_size,3), strides=(1,1), activation='tanh', padding='valid'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
#    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))

    return model

def create_eegnet(data_shape, f=1):

    timepoints = data_shape[1]
    channels = data_shape[2]

    spatial_filters = 16
    model = Sequential()

    model.add(Conv2D(spatial_filters, (1,channels), activation='relu',
                     kernel_regularizer=regularizers.l1_l2(0.0001), input_shape=(timepoints, channels, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Reshape((timepoints,spatial_filters,1)))

    model.add(Conv2D(4, (16*f,2), strides=(1,1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((4,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(4, (2*f,8), strides=(1,1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((4,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model
