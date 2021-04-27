from MB3DEEGNet import EEGNet3D
import numpy as np
import sys

import tensorflow as tf

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

init_lr = 0.0001
X = 7
Y = 6
T = 240
F1 = 8
F2 = 16
D = 2
N = 4

print("X:\t", X)
print("Y:\t", Y)
print("T:\t", T)
print("N:\t", N)
print("Learning Rate: ", init_lr)

print("Verifying thread usage params")
print("Intra Op Threads:\t", tf.config.threading.get_intra_op_parallelism_threads())
print("Inter Op Threads:\t", tf.config.threading.get_inter_op_parallelism_threads())

model = EEGNet3D(X, Y, T, F1, F2, D, N)

loss_functions = ('sparse_categorical_crossentropy', 'mean_absolute_error')

opt = tf.keras.optimizers.Adam(init_lr)

print("Model compiling")

model.compile(loss=loss_functions[0], optimizer=opt, metrics=['accuracy'])

print("Model compiled")

model.summary()

X_data, Y_data = load_data(DATA_DIR, 1, "T")
X_val, Y_val = load_data(DATA_DIR, 1, "E")
#X_2_val, Y_2_val = load_data(DATA_DIR, 2, "E")

model.fit(X_data, Y_data, epochs=10, verbose=0, callbacks=[CustomCallback()])


_, acc = model.evaluate(X_val, Y_val, verbose=0, callbacks=[CustomCallback()])

print("Accuracy: %.2f" % (acc*100))

#print("Testing alternate Subject #2")

#_, acc = model.evaluate(X_2_val, Y_2_val, verbose=0, callbacks=[CustomCallback()])

#print("Accuracy: %.2f" % (acc*100))

