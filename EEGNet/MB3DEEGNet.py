from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Softmax
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import AveragePooling3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten, Add
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

from DepthwiseConv3D import DepthwiseConv3D

AVERAGE_POOLING_SIZE1 = (1, 1, 4)

AVERAGE_POOLING_SIZE2 = (1, 1, 8)

SRF_SIZE_1 = (1, 1, 16)
MRF_SIZE_1 = (1, 1, 32)
LRF_SIZE_1 = (1, 1, 64)

SRF_SIZE_2 = (1, 1, 4)
MRF_SIZE_2 = (1, 1, 8)
LRF_SIZE_2 = (1, 1, 16)

def EEGNet3D_Branch(X, Y, T, F1, F2, D, N, size1, size2, block, name):
	block1 = Conv3D(F1, size1, padding='same', activation='linear', use_bias=False, name="Conv3D"+name)(block)
	block = BatchNormalization()(block1)
	block1 = DepthwiseConv3D((X, Y, 1), depth_multiplier=D, padding='valid', activation='linear', use_bias=False, name="DepthwiseConv3D"+name)(block1)
	block1 = BatchNormalization()(block1)
	block1 = Activation('elu')(block1)
	block1 = AveragePooling3D(AVERAGE_POOLING_SIZE1)(block1)
	block1 = Dropout(0.25)(block1)
	
	block2 = DepthwiseConv3D(size2, depth_multiplier=D, padding='same', activation='linear', use_bias=False, name="Separable-Depthwise"+name)(block1)
	block2 = Conv3D(F2, (1, 1, 1), padding='valid', activation='linear', use_bias=False, name="PointwiseConv3D-Separable"+name)(block2)
	block2 = BatchNormalization()(block2)
	block2 = Activation('elu')(block2)
	block2 = AveragePooling3D(AVERAGE_POOLING_SIZE2)(block2)
	block2 = Dropout(0.25)(block2)
	flatten = Flatten()(block2)
	dense = Dense(N, activation='softmax')(flatten)
	return dense

def EEGNet3D(X, Y, T, F1, F2, D, N):
	block = Input(shape = (X, Y, T, 1))

	branch0 = EEGNet3D_Branch(X, Y, T, F1, F2, D, N, SRF_SIZE_1, SRF_SIZE_2, block, "0")
	branch1 = EEGNet3D_Branch(X, Y, T, F1, F2, D, N, MRF_SIZE_1, MRF_SIZE_2, block, "1")
	branch2 = EEGNet3D_Branch(X, Y, T, F1, F2, D, N, LRF_SIZE_1, LRF_SIZE_2, block, "2")

	final = Add()([branch0, branch1, branch2])
	out = Softmax()(final)
	
	model = Model(inputs=block, outputs=out)
	return model
