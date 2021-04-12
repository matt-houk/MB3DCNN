from EEGNet3D import EEGNet3D
import numpy as np

from tensorflow.keras.backend import image_data_format, set_image_data_format

from tensorflow.keras.optimizers import Adam

DATA_DIR = "/share/multibranch/datasets/BCICIV_2a_cropped/"

print(image_data_format())
#set_image_data_format('channels_first')
#print(image_data_format())

def load_data(data_dir, num, file_type):
	x = np.load(data_dir + "A0" + str(num) + file_type + "D_cropped.npy").astype(np.float64)
	y = np.load(data_dir + "A0" + str(num) + file_type + "K_cropped.npy").astype(np.float64)
	return x, y

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
print("F1:\t", F1)
print("F2:\t", F2)
print("D:\t", D)
print("D:\t", N)

model = EEGNet3D(X, Y, T, F1, F2, D, N)

loss_functions = ('sparse_categorical_crossentropy', 'mean_absolute_error')

opt = Adam(0.001)

print("Model compiling")

model.compile(loss=loss_functions[0], optimizer=opt, metrics=['accuracy'])

print("Model compiled")

model.summary()

X_data, Y_data = load_data(DATA_DIR, 1, "T")

print("Input Shape:\n", X_data.shape)
print("Output Shape:\n", Y_data.shape)

model.fit(X_data, Y_data, epochs=10)

_, acc = model.evaluate(X_data, Y_data)

print("Accuracy: %.2f" % (acc*100))

