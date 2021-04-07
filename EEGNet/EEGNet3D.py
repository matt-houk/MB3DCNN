from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, SpatialDropout2D
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import AveragePooling3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

AVERAGE_POOLING_SIZE1 = (1, 1, 4)

AVERAGE_POOLING_SIZE2 = (1, 1, 8)

DATASET_FREQ = 250

TRIAL_DUR = 1.25

def EEGNet3D(nb_classes, xDim=7, yDim=6, timesteps=240, dropoutRate=0.5, size=(1, 1, int(DATASET_FREQ/2)), F1=8, F2=16, D=2, norm_rate=0.25, dropoutType = 'Dropout'):
	if dropoutType == 'Dropout':
		dropoutType = Dropout
	elif drouputType == 'SpatialDropout2D':
		dropoutType = SpatialDropout2D
	else:
		raise ValueError('dropoutType must be one of SpatialDropout2d or Dropout, passed as a string.')
	
	input1 = Input(shape = (xDim, yDim, timesteps, 1))
	
	block1 = Conv3D(F1, size, padding='same', input_shape=(xDim, yDim, timesteps, 1), use_bias = False)(input1)
	
	block1 = BatchNormalization()(block1)
	# Implementation of DepthwiseConv3D Using Conv3D class --- Investigate use of other params
	block1 = Conv3D(D*F1, (xDim, yDim, 1), groups=8, padding='same', use_bias=False, name="DepthwiseConv3D")(block1)
	block1 = BatchNormalization()(block1)
	block1 = Activation('elu')(block1)
	block1 = AveragePooling3D(AVERAGE_POOLING_SIZE1)(block1)
	block1 = dropoutType(dropoutRate)(block1)
	
	# Implementation of SeperableConv3D using Conv3D class
	block2 = Conv3D(F2, (1, 1, 16), groups=16, padding='same', use_bias=False, name="SeperableConv3D")(block1)
	block2 = BatchNormalization()(block2)
	block2 = Activation('elu')(block2)
	block2 = AveragePooling3D(AVERAGE_POOLING_SIZE2)(block2)
	block2 = dropoutType(dropoutRate)(block2)
	flatten = Flatten(name = 'flatten')(block2)
	dense = Dense(nb_classes, name = 'dense')(flatten)
	softmax = Activation('softmax', name = 'softmax')(dense)
	return Model(inputs=input1, outputs=softmax)


