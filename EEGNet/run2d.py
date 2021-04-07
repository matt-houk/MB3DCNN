from EEGNet2D import EEGNet2D
import numpy as np

from keras.optimizers import Adam

DATA_DIR = "../datasets/BCICIV_2a_2d_processed/"

def load_data(data_dir, num, file_type):
	x = np.load(data_dir + "A0" + str(num) + file_type + "D_processed.npy").astype(np.float32)
	y = np.load(data_dir + "A0" + str(num) + file_type + "K_processed.npy").astype(np.float32)
	return x, y

model = EEGNet2D(22, 313, 8, 16, 2, 4)

loss_functions = ('sparse_categorical_crossentropy', 'mean_absolute_error')

opt = Adam(0.001)

model.compile(loss=loss_functions[0], optimizer=opt, metrics=['accuracy'])

model.summary()

X, Y = load_data(DATA_DIR, 1, "T")

model.fit(X, Y, epochs=10)

_, acc = model.evaluate(X, Y)

print("Accuracy: %.2f" % (acc*100))
