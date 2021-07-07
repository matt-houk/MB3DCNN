from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Softmax
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import AveragePooling3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten, Add
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

REG = l2(0.00001)

def EEGNet3D(nb_classes, XDim = 7, YDim = 6, Samples = 240, dropoutRate = 0.5, kernLength = 64, F1 = 8, D = 2, F2 = 16, norm_rate = 0.15, dropoutType = 'Dropout', branches = ("SRF", "MRF", "LRF")):
	if dropoutType == 'SpatialDropout3D':
		dropoutType = SpatialDropout3D
	elif dropoutType == 'Dropout':
		dropoutType = Dropout
	else:
		raise ValueError('dropoutType must be one of SpatialDropout3D or Dropout, passed as a string.')
    
	input1 = Input(shape = (XDim, YDim, Samples, 1))
	
	add_params = []
	if ("SRF" in branches):
		SRF_branch = EEGNet3D_Branch(nb_classes, XDim, YDim, Samples, dropoutRate, int(kernLength/2), F1, D, F2, norm_rate, dropoutType, input1)
		add_params.append(SRF_branch)
	if ("MRF" in branches):
		MRF_branch = EEGNet3D_Branch(nb_classes, XDim, YDim, Samples, dropoutRate, int(kernLength), F1, D, F2, norm_rate, dropoutType, input1)
		add_params.append(MRF_branch)
	if ("LRF" in branches):
		LRF_branch = EEGNet3D_Branch(nb_classes, XDim, YDim, Samples, dropoutRate, int(kernLength*2), F1, D, F2, norm_rate, dropoutType, input1)
		add_params.append(LRF_branch)

	final = Add()(add_params)

	softmax = Activation('softmax', name = 'softmax')(final)
        
	return Model(inputs=input1, outputs=softmax)

def EEGNet3D_Branch(nb_classes, XDim, YDim, Samples, dropoutRate, kernLength, F1, D, F2, norm_rate, dropoutType, block):
	block1 = Conv3D(F1, (1, 1, kernLength), padding = 'same', input_shape = (XDim, YDim, Samples, 1), use_bias = False)(block)
	block1 = BatchNormalization()(block1)
	block1 = Conv3D(D*F1, (XDim, YDim, 1), groups = F1, kernel_constraint = max_norm(1.), use_bias = False)(block1)
	block1 = BatchNormalization()(block1)
	block1 = Activation('elu')(block1)
	block1 = AveragePooling3D((1, 1, 4))(block1)
	block1 = dropoutType(dropoutRate)(block1)

	block2 = Conv3D(F2, (1, 1, 16), groups = F2, use_bias = False, padding = 'same')(block1)
	block2 = Conv3D(F2, (1, 1, 1), use_bias = False, padding = 'same')(block2) 
	block2 = BatchNormalization()(block2)
	block2 = Activation('elu')(block2)
	block2 = AveragePooling3D((1, 1, 8))(block2)
	block2 = dropoutType(dropoutRate)(block2)

	flatten = Flatten()(block2)

	return Dense(nb_classes, kernel_constraint = max_norm(norm_rate))(flatten)
