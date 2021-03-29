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
from sklearn.model_selection import StratifiedKFold

from DonghyunMBCNN import MultiBranchCNN

# Global Variables

# The directory of the process data, must have been converted and cropped, reference dataProcessing.py and crop.py
DATA_DIR = "../datasets/BCICIV_2a_cropped/"
# Which trial subject will be trained
SUBJECT = 1

# The number of classification categories, for motor imagery, there are 4
NUM_CLASSES = 4
# The number of timesteps in each input array
TIMESTEPS = 240
# The X-Dimension of the dataset
XDIM = 7
# The Y-Dimension of the dataset
YDIM = 6
# The delta loss requirement for lower training rate
LOSS_THRESHOLD = 0.01
# Initial learning rate for ADAM optimizer
INIT_LR = 0.01
# Define Which NLL (Negative Log Likelihood) Loss function to use, either "NLL1", "NLL2", or "SCCE"
LOSS_FUNCTION = 'NLL2'
# Defines which optimizer is in use, either "ADAM" or "SGD"
OPTIMIZER = 'SGD'
# Whether training output should be given
VERBOSE = 1
# Determines whether K-Fold Cross Validation is used
USE_KFOLD = False
# Number of ksplit validation, must be atleast 2
KFOLD_NUM = 2
# Specifies which model structure will be used, '1' corresponds to the Create_Model function and '2' corresponds to Donghyun's model.
USE_STRUCTURE = '2'

# Number of epochs to train for
EPOCHS = 10

# Receptive field sizes
SRF_SIZE = (2, 2, 1)
MRF_SIZE = (2, 2, 3)
LRF_SIZE = (2, 2, 5)

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
	x = np.load(data_dir + "A0" + str(num) + "TD_cropped.npy").astype(np.float32)
	y = np.load(data_dir + "A0" + str(num) + "TK_cropped.npy").astype(np.float32)
	return x, y

def create_receptive_field(size, strides, model, name):
	modelRF = Conv3D(kernel_size = size, strides=strides, filters=32, padding='same', name=name+'1')(model)
	modelRF1 = BatchNormalization()(modelRF)
	modelRF2 = Activation('elu')(modelRF1)

	modelRF3 = Conv3D(kernel_size = size, strides=strides, filters=64, padding='same', name=name+'2')(modelRF2)
	modelRF4 = BatchNormalization()(modelRF3)
	modelRF5 = Activation('elu')(modelRF4)

	modelRF6 = Flatten()(modelRF5)

	modelRF7 = Dense(32)(modelRF6)
	modelRF8 = BatchNormalization()(modelRF7)
	modelRF9 = Activation('relu')(modelRF8)

	modelRF10 = Dense(32)(modelRF9)
	modelRF11 = BatchNormalization()(modelRF10)
	modelRF12 = Activation('relu')(modelRF11)
	return Dense(NUM_CLASSES, activation='softmax')(modelRF12)

def Create_Model():
	# Model Creation

	model1 = Input(shape=(1, XDIM, YDIM, TIMESTEPS))

	# 1st Convolution Layer
	model1a = Conv3D(kernel_size = (3, 3, 5), strides = (2, 2, 4), filters=16, name="Conv1")(model1)
	model1b = BatchNormalization()(model1a)
	model1c = Activation('elu')(model1b)

	# Small Receptive Field (SRF)

	modelSRF = create_receptive_field(SRF_SIZE, SRF_STRIDES, model1c, 'SRF')
	
	# Medium Receptive Field (MRF)

	modelMRF = create_receptive_field(MRF_SIZE, MRF_STRIDES, model1c, 'MRF')

	# Large Receptive Field (LRF)
	
	modelLRF = create_receptive_field(LRF_SIZE, LRF_STRIDES, model1c, 'LRF')

	# Add the layers - This sums each layer
	final = Add()([modelSRF, modelMRF, modelLRF])
	out = Softmax()(final)

	model = Model(inputs=model1, outputs=out)

	return model

if (LOSS_FUNCTION == 'NLL1'):
	loss_function = Loss_FN1
elif (LOSS_FUNCTION == 'NLL2'):
	loss_function = Loss_FN2
elif (LOSS_FUNCTION == 'SCCE'):
	loss_function = 'sparse_categorical_crossentropy'

# Optimizer is given as ADAM with an initial learning rate of 0.01
if (OPTIMIZER == 'ADAM'):
	opt = Adam(learning_rate = INIT_LR)
elif (OPTIMIZER == 'SGD'):
	opt = SGD(learning_rate = INIT_LR)

X, Y = load_data(DATA_DIR, SUBJECT)

if (USE_KFOLD):
	seed = 4
	kfold = StratifiedKFold(n_splits=KFOLD_NUM, shuffle=True, random_state=seed)
	cvscores = []

	for train, test in kfold.split(X, Y):
		if (USE_STRUCTURE == '1'):
			MRF_model = Create_Model()
		elif (USE_STRUCTURE == '2'):
			MRF_model = MultiBranchCNN(TIMESTEPS, YDIM, XDIM, NUM_CLASSES)
		# Compiling the model with the negative log likelihood loss function, ADAM optimizer
		MRF_model.compile(loss=loss_function, optimizer=opt, metrics=['accuracy'])

		# Training for 30 epochs
		MRF_model.fit(X[train], Y[train], epochs=30, verbose=VERBOSE)

		# Evaluating the effectiveness of the model
		scores = MRF_model.evaluate(X[test], Y[test], verbose=VERBOSE)
		print("%s: %.2f%%" % (MRF_model.metrics_names[1], scores[1]*100))
		cvscores.append(scores[1]*100)

	print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

else:
	if (USE_STRUCTURE == '1'):
		MRF_model = Create_Model()
	elif (USE_STRUCTURE == '2'):
		MRF_model = MultiBranchCNN(TIMESTEPS, YDIM, XDIM, NUM_CLASSES)

	MRF_model.compile(loss=loss_function, optimizer=opt, metrics=['accuracy'])
	
	MRF_model.fit(X, Y, epochs=EPOCHS, verbose=VERBOSE)

	_, acc = MRF_model.evaluate(X, Y, verbose=VERBOSE)

	print("Accuracy: %.2f" % (acc*100))
