from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Conv2D, BatchNormalization, Activation, AveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import DepthwiseConv2D, SeparableConv2D

def EEGNet2D(C, T, F1, F2, D, N):
	input1 = Input(shape=(C, T, 1))
	#block1 = Reshape((1, C, T))(input1)
	block2 = Conv2D(F1, (1, 64), padding='same', activation='linear', use_bias=False, name="Conv2D")(input1)
	block3 = BatchNormalization()(block2)
	block4 = Conv2D(D*F1, (C, 1), groups=F1, kernel_constraint=max_norm(1), padding='valid', activation='linear', use_bias=False, name="DepthwiseConv2D")(block3)
	block5 = BatchNormalization()(block4)
	block6 = Activation('elu')(block5)
	block7 = AveragePooling2D((4, 1))(block6)
	block8 = Dropout(0.25)(block7)
	block9 = Conv2D(F2, (1, 16), groups=int(F2/D), padding='same', activation='linear', use_bias=False, name="SeparableConv2D")(block8)
	block10 = BatchNormalization()(block9)
	block11 = Activation('elu')(block10)
	block12 = AveragePooling2D((8, 1))(block11)
	block13 = Dropout(0.25)(block12)
	block14 = Flatten()(block13)
	block15 = Dense(N, activation='softmax', kernel_constraint=max_norm(0.25))(block14)
	return Model(inputs=input1, outputs=block15)
