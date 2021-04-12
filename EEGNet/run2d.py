from EEGNet2D import EEGNet2D
import numpy as np

from tensorflow.keras.backend import image_data_format, set_image_data_format

from tensorflow.keras.optimizers import Adam

DATA_DIR = "/share/multibranch/datasets/BCICIV_2a_2d_processed/"

print(image_data_format())
set_image_data_format('channels_first')
print(image_data_format())

def load_data(data_dir, num, file_type):
	x = np.load(data_dir + "A0" + str(num) + file_type + "D_processed.npy").astype(np.float32)
	y = np.load(data_dir + "A0" + str(num) + file_type + "K_processed.npy").astype(np.float32)
	return x, y

C = 22
T = 313
F1 = 8
F2 = 16
D = 2
N = 4

print("C:\t", C)
print("T:\t", T)
print("F1:\t", F1)
print("F2:\t", F2)
print("D:\t", D)
print("N:\t", N)

model = EEGNet2D(C, T, F1, F2, D, N)

loss_functions = ('sparse_categorical_crossentropy', 'mean_absolute_error')

opt = Adam(0.001)

print("Model compiling")

model.compile(loss=loss_functions[0], optimizer=opt, metrics=['accuracy'])

print("Model compiled")

model.summary()

X, Y = load_data(DATA_DIR, 1, "T")

model.fit(X, Y, epochs=10)

"""

_, acc = model.evaluate(X, Y)

print("Accuracy: %.2f" % (acc*100))
"""
