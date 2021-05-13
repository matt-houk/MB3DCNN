from EEGNet3D import EEGNet3D
import numpy as np
import sys

from tensorflow.keras.backend import image_data_format, set_image_data_format
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam

from tensorflow.config.threading import set_intra_op_parallelism_threads, set_inter_op_parallelism_threads

set_intra_op_parallelism_threads(2)
set_inter_op_parallelism_threads(2)

DATA_DIR = "/share/multibranch/datasets/BCICIV_2a_cropped/"

print(image_data_format())
#set_image_data_format('channels_first')
#print(image_data_format())

def load_data(data_dir, num, file_type):
	x = np.load(data_dir + "A0" + str(num) + file_type + "D_cropped.npy").astype(np.float64)
	y = np.load(data_dir + "A0" + str(num) + file_type + "K_cropped.npy").astype(np.float64)
	return x, y

class CustomCallback(Callback):
	def on_epoch_end(self, epoch, logs=None):
		print("Finished epoch {} with Loss: {} and Accuracy: {}".format(epoch, logs['loss'], logs['accuracy']))

init_lr = float(sys.argv[1])
F1 = int(sys.argv[2])
F2 = int(sys.argv[3])
D = int(sys.argv[4])

X = 7
Y = 6
T = 240
N = 4

print("X:\t", X)
print("Y:\t", Y)
print("T:\t", T)
print("F1:\t", F1)
print("F2:\t", F2)
print("D:\t", D)
print("N:\t", N)
print("Learning Rate: ", init_lr)

model = EEGNet3D(X, Y, T, F1, F2, D, N)

loss_functions = ('sparse_categorical_crossentropy', 'mean_absolute_error')

opt = Adam(init_lr)

print("Model compiling")

model.compile(loss=loss_functions[0], optimizer=opt, metrics=['accuracy'])

print("Model compiled")

model.summary()

X_data, Y_data = load_data(DATA_DIR, 1, "T")
X_val, Y_val = load_data(DATA_DIR, 1, "E")

model.fit(X_data, Y_data, epochs=10, verbose=0, callbacks=[CustomCallback()])

_, acc = model.evaluate(X_data, Y_data, verbose=0, callbacks=[CustomCallback()])

print("Accuracy: %.2f" % (acc*100))

