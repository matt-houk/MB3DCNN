from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Softmax
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import AveragePooling3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten, Add
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

from DepthwiseConv3D import DepthwiseConv3D

AVERAGE_POOLING_SIZE1 = (1, 1, 1)

AVERAGE_POOLING_SIZE2 = (1, 1, 1)

DROPOUT_VAL = 0.5

REG = l2(0.0001)

def EEGNet3D_Branch(X, Y, T, F1, F2, D, N, size1, size2, block, name):
	block1 = DepthwiseConv3D((X, Y, 1), depth_multiplier=D, kernel_regularizer=REG, padding='valid', activation='linear', use_bias=False, name="DepthwiseConv3D"+name)(block)
	block1 = BatchNormalization()(block1)
	block1 = Activation('elu')(block1)
	block1 = AveragePooling3D(AVERAGE_POOLING_SIZE1)(block1)
	block1 = Dropout(DROPOUT_VAL)(block1)
	
	block2 = DepthwiseConv3D(size2, depth_multiplier=D, kernel_regularizer=REG, padding='same', activation='linear', use_bias=False, name="Separable-Depthwise"+name)(block1)
	block2 = Conv3D(F2, (1, 1, 1), padding='valid', kernel_regularizer=REG, activation='linear', use_bias=False, name="PointwiseConv3D-Separable"+name)(block2)
	block2 = BatchNormalization()(block2)
	block2 = Activation('elu')(block2)
	block2 = AveragePooling3D(AVERAGE_POOLING_SIZE2)(block2)
	block2 = Dropout(DROPOUT_VAL)(block2)
	flatten = Flatten()(block2)
	dense = Dense(N, activation='softmax')(flatten)
	return dense

def EEGNet3D(X, Y, T, F1, F2, D, N, size1, size2, sf):
	block_in = Input(shape = (X, Y, T, 1))
	block = Conv3D(F1, (1, 1, size1), kernel_regularizer=REG, padding='same', activation='linear', use_bias=False, name="Conv3D1")(block_in)
	block = BatchNormalization()(block)
	
	branch0 = EEGNet3D_Branch(X, Y, T, F1, F2, D, N, (1, 1, size1), (1, 1, size2), block, "0")
	branch1 = EEGNet3D_Branch(X, Y, T, F1*sf, F2*sf, D*sf, N, (1, 1, size1*sf), (1, 1, size2*sf), block, "1")
	branch2 = EEGNet3D_Branch(X, Y, T, F1*sf*sf, F2*sf*sf, D*sf*sf, N, (1, 1, size1*sf*sf), (1, 1, size2*sf*sf), block, "2")

	final = Add()([branch0, branch1, branch2])
	out = Softmax()(final)
	
	model = Model(inputs=block_in, outputs=out)
	return model
