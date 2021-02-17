"""
Matthew Houk

Working to recreate results of paper listed in readme on behalf of NCSU BCI Lab


"""

import os
import gc

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from keras import layers
from keras import backend as K
from keras import regularizers
from keras.constraints import max_norm
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
from keras.models import Model, clone_model
from keras.initializers import glorot_uniform
from keras.layers import Conv3D, Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout, Add, Softmax, Concatenate

import matplotlib.pyplot as plt

# Global Variables

NUM_CLASSES = 5
TIMESTEPS = 10

SRF_SIZE = (2, 2, 1)
MRF_SIZE = (2, 2, 3)
LRF_SIZE = (2, 2, 5)

SRF_STRIDES = (2, 2, 1)
MRF_STRIDES = (2, 2, 2)
LRF_STRIDES = (2, 2, 4)


def Create_Model():
	# Model Creation

	model1 = Input(shape=(6, 7, TIMESTEPS, 1))
	model1a = Conv3D(kernel_size = (3, 3, 5), strides = (2, 2, 4), filters=16, name="Conv1")(model1)
	model1b = BatchNormalization()(model1a)
	model1c = Activation('elu')(model1b)

	# Small Receptive Field (SRF)

	modelsrf = Conv3D(kernel_size = SRF_SIZE, strides = SRF_STRIDES, filters=32, padding='same', name='SRF1')(model1c)
	modelsrf1 = BatchNormalization()(modelsrf)
	modelsrf2 = (Activation('elu'))(modelsrf1)

	modelsrf3 = Conv3D(kernel_size = SRF_SIZE, strides = SRF_STRIDES, filters=64, padding='same', name='SRF2')(modelsrf2)
	modelsrf4 = BatchNormalization()(modelsrf3)
	modelsrf5 = Activation('elu')(modelsrf4)

	modelsrf6 = Flatten()(modelsrf5)

	modelsrf7 = Dense(32)(modelsrf6)
	modelsrf8 = BatchNormalization()(modelsrf7)
	modelsrf9 = Activation('relu')(modelsrf8)

	modelsrf10 = Dense(32)(modelsrf9)
	modelsrf11 = BatchNormalization()(modelsrf10)
	modelsrf12 = Activation('relu')(modelsrf11)

	modelsrf13 = Dense(NUM_CLASSES)(modelsrf12)
	modelsrf_final = Softmax()(modelsrf13)


	# Medium Receptive Field (MRF)

	modelmrf = Conv3D(kernel_size = MRF_SIZE, strides = MRF_STRIDES, filters=32, padding='same', name='MRF1')(model1c)
	modelmrf1 = BatchNormalization()(modelmrf)
	modelmrf2 = Activation('elu')(modelmrf1)

	modelmrf3 = Conv3D(kernel_size = MRF_SIZE, strides = MRF_STRIDES, filters=64, padding='same', name='MRF2')(modelmrf2)
	modelmrf4 = BatchNormalization()(modelmrf3)
	modelmrf5 = Activation('elu')(modelmrf4)

	modelmrf6 = Flatten()(modelmrf5)

	modelmrf7 = Dense(32)(modelmrf6)
	modelmrf8 = BatchNormalization()(modelmrf7)
	modelmrf9 = Activation('relu')(modelmrf8)

	modelmrf10 = Dense(32)(modelmrf9)
	modelmrf11 = BatchNormalization()(modelmrf10)
	modelmrf12 = Activation('relu')(modelmrf11)

	modelmrf13 = Dense(NUM_CLASSES)(modelmrf12)
	modelmrf_final = Softmax()(modelmrf13)

	# Large Receptive Field (LRF)

	modellrf = Conv3D(kernel_size = LRF_SIZE, strides = LRF_STRIDES, filters=32, padding='same', name='LRF1')(model1c)
	modellrf1 = BatchNormalization()(modellrf)
	modellrf2 = Activation('elu')(modellrf1)

	modellrf3 = Conv3D(kernel_size = LRF_SIZE, strides = LRF_STRIDES, filters=64, padding='same', name='LRF2')(modellrf2)
	modellrf4 = BatchNormalization()(modellrf3)
	modellrf5 = Activation('elu')(modellrf4)

	modellrf6 = Flatten()(modellrf5)

	modellrf7 = Dense(32)(modellrf6)
	modellrf8 = BatchNormalization()(modellrf7)
	modellrf9 = Activation('relu')(modellrf8)

	modellrf10 = Dense(32)(modellrf9)
	modellrf11 = BatchNormalization()(modellrf10)
	modellrf12 = Activation('relu')(modellrf11)

	modellrf13 = Dense(NUM_CLASSES)(modellrf12)
	modellrf_final = Softmax()(modellrf13)

	final = Add()([modelsrf_final, modelmrf_final, modellrf_final])
	out = Softmax()(final)

	model = Model(inputs=model1, outputs=out)

	return model

MRF_model = Create_Model()

MRF_model.compile()
MRF_model.summary()
