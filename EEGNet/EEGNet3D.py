from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, SpatialDropout2D
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import AveragePooling3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

from DepthwiseConv3D import DepthwiseConv3D

AVERAGE_POOLING_SIZE1 = (1, 1, 4)

AVERAGE_POOLING_SIZE2 = (1, 1, 8)

def EEGNet3D(X, Y, T, F1, F2, D, N):
	input1 = Input(shape = (X, Y, T, 1))
	block1 = Conv3D(F1, (1, 1, 64), padding='same', activation='linear', use_bias=False, name="Conv3D")(input1)
	
	block1 = BatchNormalization()(block1)
	# Implementation of DepthwiseConv3D Using Conv3D class --- Investigate use of other params
	#block1 = Conv3D(D*F1, (X, Y, 1), groups=8, padding='valid', activation='linear', use_bias=False, name="DepthwiseConv3D")(block1)
	block1 = DepthwiseConv3D((X, Y, 1), depth_multiplier=D, padding='valid', activation='linear', use_bias=False, name="DepthwiseConv3D")(block1)
	block1 = BatchNormalization()(block1)
	block1 = Activation('elu')(block1)
	block1 = AveragePooling3D(AVERAGE_POOLING_SIZE1)(block1)
	block1 = Dropout(0.25)(block1)
	
	# Implementation of SeperableConv3D using Conv3D class
	#block2 = Conv3D(F2, (1, 1, 16), groups=F2, padding='same', activation='linear', use_bias=False, name="DepthwiseConv3D-Separable")(block1)
	block2 = DepthwiseConv3D((1, 1, 16), padding='same', activation='linear', use_bias=False, name="Separable-Depthwise")(block1)
	block2 = Conv3D(F2, (1, 1, 1), padding='valid', activation='linear', use_bias=False, name="PointwiseConv3D-Separable")(block2)
	block2 = BatchNormalization()(block2)
	block2 = Activation('elu')(block2)
	block2 = AveragePooling3D(AVERAGE_POOLING_SIZE2)(block2)
	block2 = Dropout(0.25)(block2)
	flatten = Flatten(name = 'flatten')(block2)
	dense = Dense(N, activation='softmax', name = 'dense')(flatten)
	return Model(inputs=input1, outputs=dense)


