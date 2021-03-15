"""
Matthew Houk

Working to recreate results of paper listed in readme on behalf of NCSU BCI Lab

"""


# Filters out warnings
import warnings
warnings.filterwarnings("ignore")

# Imports, as of 3/10, all are necessary
import numpy as np
import tensorflow as tf
from keras import layers
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import Callback
from keras.layers import Conv3D, Input, Dense, Activation, BatchNormalization, Flatten, Add, Softmax

# Global Variables

# The directory of the process data, must have been converted and cropped, reference dataProcessing.py and crop.py
DATA_DIR = "../datasets/BCICIV_2a_processed/"

# The number of classification categories, for motor imagery, there are 4
NUM_CLASSES = 4
# The number of timesteps in each input array
TIMESTEPS = 240
# The delta loss requirement for lower training rate
LOSS_THRESHOLD = 0.01
# Initial learning rate for ADAM optimizer
INIT_LR = 0.005
# Define Which NLL (Negative Log Likelihood) Loss function to use, either "NLL1" or "NLL2"
LOSS_FUNCTION = 'NLL2'

# Receptive field sizes
SRF_SIZE = (2, 2, 1)
MRF_SIZE = (2, 2, 4)
LRF_SIZE = (2, 2, 7)

# Strides for each receptive field
SRF_STRIDES = (2, 2, 1)
MRF_STRIDES = (2, 2, 2)
LRF_STRIDES = (2, 2, 4)

# This is meant to handle the reduction of the learning rate, current is not accurate, I have been unable to access the loss information from each Epoch
# The expectation is that if the delta loss is < threshold, learning rate *= 0.1. Threshold has not been set yet.
class LearningRateReducerCb(Callback):
	def __init__(self):
		self.history = {}
	def on_epoch_end(self, epoch, logs={}):

		for k, v in logs.items():
			self.history.setdefault(k, []).append(v)
				
		fin_index = len(self.history['loss']) - 1
		if (fin_index >= 1):
			if (self.history['loss'][fin_index-1] - self.history['loss'][fin_index] > LOSS_THRESHOLD):
				old_lr = self.model.optimizer.lr.read_value()
				new_lr = old_lr*0.1
				print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
				self.model.optimizer.lr.assign(new_lr)	

# The Negative Log Likelihood function
def Loss_FN1(y_true, y_pred, sample_weight=None):
	return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1) # This is another loss function that I tried, was less effective

# Second NLL function, generally seems to work better
def Loss_FN2(y_true, y_pred, sample_weight=None):
	n_dims = int(int(y_pred.shape[1])/2)
	mu = y_pred[:, 0:n_dims]
	logsigma = y_pred[:, n_dims:]
	mse = -0.5*K.sum(K.square((y_true-mu)/K.exp(logsigma)), axis=1)
	sigma_trace = -K.sum(logsigma, axis=1)
	log2pi = -0.5*n_dims*np.log(2*np.pi)
	log_likelihood = mse+sigma_trace+log2pi
	return K.mean(-log_likelihood)

	
# Loads given data into two arrays, x and y, while also ensuring that all values are formatted as float32s
def load_data(data_dir, num):
	x = np.load(data_dir + "A0" + str(num) + "T_cropped_shuffled.npy").astype(np.float32)
	y = np.load(data_dir + "A0" + str(num) + "TK_cropped_shuffled.npy").astype(np.float32)
	return x, y

def Create_Model():
	# Model Creation

	model1 = Input(shape=(7, 6, TIMESTEPS, 1))

	# 1st Convolution Layer
	model1a = Conv3D(kernel_size = (3, 3, 5), strides = (2, 2, 4), filters=16, name="Conv1")(model1)
	model1b = BatchNormalization()(model1a)
	model1c = Activation('elu')(model1b)

	# Small Receptive Field (SRF)

	# 2nd Convolution Layer
	modelsrf = Conv3D(kernel_size = SRF_SIZE, strides = SRF_STRIDES, filters=32, padding='same', name='SRF1')(model1c)
	modelsrf1 = BatchNormalization()(modelsrf)
	modelsrf2 = (Activation('elu'))(modelsrf1)

	# 3rd Convolution Layer
	modelsrf3 = Conv3D(kernel_size = SRF_SIZE, strides = SRF_STRIDES, filters=64, padding='same', name='SRF2')(modelsrf2)
	modelsrf4 = BatchNormalization()(modelsrf3)
	modelsrf5 = Activation('elu')(modelsrf4)

	# Flatten
	modelsrf6 = Flatten()(modelsrf5)

	# Dense Layer
	modelsrf7 = Dense(32)(modelsrf6)
	modelsrf8 = BatchNormalization()(modelsrf7)
	modelsrf9 = Activation('relu')(modelsrf8)

	# Dense Layer
	modelsrf10 = Dense(32)(modelsrf9)
	modelsrf11 = BatchNormalization()(modelsrf10)
	modelsrf12 = Activation('relu')(modelsrf11)

	# Dense Layer
	modelsrf_final = Dense(NUM_CLASSES, activation='softmax')(modelsrf12)
	#modelsrf_final = Softmax()(modelsrf13)


	# Medium Receptive Field (MRF)

	# 2nd Convolution Layer
	modelmrf = Conv3D(kernel_size = MRF_SIZE, strides = MRF_STRIDES, filters=32, padding='same', name='MRF1')(model1c)
	modelmrf1 = BatchNormalization()(modelmrf)
	modelmrf2 = Activation('elu')(modelmrf1)

	# 3rd Convolution Layer
	modelmrf3 = Conv3D(kernel_size = MRF_SIZE, strides = MRF_STRIDES, filters=64, padding='same', name='MRF2')(modelmrf2)
	modelmrf4 = BatchNormalization()(modelmrf3)
	modelmrf5 = Activation('elu')(modelmrf4)

	# Flatten
	modelmrf6 = Flatten()(modelmrf5)

	# Dense Layer
	modelmrf7 = Dense(32)(modelmrf6)
	modelmrf8 = BatchNormalization()(modelmrf7)
	modelmrf9 = Activation('relu')(modelmrf8)

	# Dense Layer
	modelmrf10 = Dense(32)(modelmrf9)
	modelmrf11 = BatchNormalization()(modelmrf10)
	modelmrf12 = Activation('relu')(modelmrf11)

	# Dense Layer
	modelmrf_final = Dense(NUM_CLASSES, activation='softmax')(modelmrf12)
	#modelmrf_final = Softmax()(modelmrf13)

	# Large Receptive Field (LRF)

	# 2nd Convolution Layer
	modellrf = Conv3D(kernel_size = LRF_SIZE, strides = LRF_STRIDES, filters=32, padding='same', name='LRF1')(model1c)
	modellrf1 = BatchNormalization()(modellrf)
	modellrf2 = Activation('elu')(modellrf1)

	# 3rd Convolution Layer
	modellrf3 = Conv3D(kernel_size = LRF_SIZE, strides = LRF_STRIDES, filters=64, padding='same', name='LRF2')(modellrf2)
	modellrf4 = BatchNormalization()(modellrf3)
	modellrf5 = Activation('elu')(modellrf4)

	# Flatten
	modellrf6 = Flatten()(modellrf5)

	# Dense Layer
	modellrf7 = Dense(32)(modellrf6)
	modellrf8 = BatchNormalization()(modellrf7)
	modellrf9 = Activation('relu')(modellrf8)

	# Dense Layer
	modellrf10 = Dense(32)(modellrf9)
	modellrf11 = BatchNormalization()(modellrf10)
	modellrf12 = Activation('relu')(modellrf11)

	# Dense Layer
	modellrf_final = Dense(NUM_CLASSES, activation='softmax')(modellrf12)
	#modellrf_final = Softmax()(modellrf13)

	# Add the layers - This sums each layer
	final = Add()([modelsrf_final, modelmrf_final, modellrf_final])
	out = Softmax()(final)

	model = Model(inputs=model1, outputs=out)

	return model

MRF_model = Create_Model()

if (LOSS_FUNCTION == 'NLL1'):
	loss_function = Loss_FN1
elif (LOSS_FUNCTION == 'NLL2'):
	loss_function = Loss_FN2

# Optimizer is given as ADAM with an initial learning rate of 0.01
opt = Adam(learning_rate = INIT_LR)
# Compiling the model with the negative log likelihood loss function, ADAM optimizer
MRF_model.compile(loss=loss_function, optimizer=opt, metrics=['accuracy'])
MRF_model.summary()

X, Y = load_data(DATA_DIR, 1)

# Training for 30 epochs
MRF_model.fit(X, Y, callbacks=[LearningRateReducerCb()], epochs=30)

# Evaluating the effectiveness of the model
_, acc = MRF_model.evaluate(X, Y)

print("Accuracy: %.2f" % (acc*100))
