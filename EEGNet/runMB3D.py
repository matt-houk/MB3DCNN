from MB3DEEGNet import EEGNet3D
import numpy as np
import sys
import os

try:
	print("TF_NUM_INTEROP_THREADS =", os.environ['TF_NUM_INTEROP_THREADS'])
except KeyError:
	print("'TF_NUM_INTEROP_THREADS' not found, adding and setting to '1'")
	os.environ['TF_NUM_INTEROP_THREADS'] = '1'
try:
	print("TF_NUM_INTRAOP_THREADS =", os.environ['TF_NUM_INTRAOP_THREADS'])
except KeyError:
	print("'TF_NUM_INTRAOP_THREADS' not found, adding and setting to '1'")
	os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

OUT_FILE = "./batch_data/oat.csv"

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

physical_devices = tf.config.list_physical_devices('GPU')
try:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
	print("Invalid device or cannot modify virtual device once initialized")
	pass

DATA_DIR = "/share/multibranch/datasets/BCICIV_2a_cropped/"

def load_data(data_dir, num, file_type):
	x = np.load(data_dir + "A0" + str(num) + file_type + "D_cropped.npy").astype(np.float64)
	y = np.load(data_dir + "A0" + str(num) + file_type + "K_cropped.npy").astype(np.float64)
	return x, y

class CustomCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		print("Finished epoch {} with Loss: {} and Accuracy: {}".format(epoch, logs['loss'], logs['accuracy']))

tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)

try:
	print("TF_NUM_INTEROP_THREADS =", os.environ['TF_NUM_INTEROP_THREADS'])
except KeyError:
	print("'TF_NUM_INTEROP_THREADS' not found, adding and setting to '1'")
	os.environ['TF_NUM_INTEROP_THREADS'] = '1'
try:
	print("TF_NUM_INTRAOP_THREADS =", os.environ['TF_NUM_INTRAOP_THREADS'])
except KeyError:
	print("'TF_NUM_INTRAOP_THREADS' not found, adding and setting to '1'")
	os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

X = 7
Y = 6
T = 240
N = 4

Ds = [2, 3, 4]
F1s = [2, 4, 8]
F2s = [4, 8, 16]
LRs = [0.00001, 0.0001, 0.001]
S1s = [2, 4, 8]
S2s = [1, 2, 4]
SFs = [2, 2, 2]

trialNum = int(sys.argv[1])

with open("./oat18x7x3.csv", "r") as f:
	f.readline()
	for i in range(trialNum):
		params = f.readline().strip('\r\n').split(',')
	print("Running trial #:", params[0])
	F1 = F1s[int(params[1])-1]
	F2 = F2s[int(params[2])-1]
	D = Ds[int(params[3])-1]
	init_lr = LRs[int(params[4])-1]
	size1 = S1s[int(params[5])-1]
	size2 = S2s[int(params[6])-1]
	sf = SFs[int(params[7])-1]
	
print("X:\t", X)
print("Y:\t", Y)
print("T:\t", T)
print("N:\t", N)
print("Learning Rate: ", init_lr)

print("Verifying thread usage params")
print("Intra Op Threads:\t", tf.config.threading.get_intra_op_parallelism_threads())
print("Inter Op Threads:\t", tf.config.threading.get_inter_op_parallelism_threads())


loss_functions = ('sparse_categorical_crossentropy', 'mean_absolute_error')

opt = tf.keras.optimizers.Adam(init_lr)

X_data_1, Y_data_1 = load_data(DATA_DIR, 1, "T")
X_data_2, Y_data_2 = load_data(DATA_DIR, 2, "T")
#X_2_val, Y_2_val = load_data(DATA_DIR, 2, "E")

X_data = np.concatenate([X_data_1, X_data_2])
Y_data = np.concatenate([Y_data_1, Y_data_2])

seed = 4

print("Starting K-Fold")
i = 1
cvscores = []
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
for train, test in kfold.split(X_data, Y_data):
	print("Performing K-Fold", i)
	model = EEGNet3D(X, Y, T, F1, F2, D, N, size1, size2, sf)
	model.compile(loss=loss_functions[0], optimizer=opt, metrics=['accuracy'])
	model.fit(X_data[train], Y_data[train], epochs=20, verbose=0, callbacks=[CustomCallback()])
	scores = model.evaluate(X_data[test], Y_data[test], verbose=0, callbacks=[CustomCallback()])
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1]*100)
	i += 1

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
acc = np.mean(cvscores)
std = np.std(cvscores)
 
with open(OUT_FILE, "a") as f:
	new_entry = "{},{},{},{},{},(1, 1, {}),(1, 1, {}),{},{},{}\n".format(trialNum, F1, F2, D, init_lr, size1, size2, sf, acc, std)
	f.write(new_entry)

print("Accuracy: %.2f" % (acc*100))

#print("Testing alternate Subject #2")

#_, acc = model.evaluate(X_2_val, Y_2_val, verbose=0, callbacks=[CustomCallback()])

#print("Accuracy: %.2f" % (acc*100))

