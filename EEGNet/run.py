from EEGNet3D import EEGNet3D
import numpy as np

DATA_DIR = "../datasets/BCICIV_2a_cropped/"

def load_data(data_dir, num, file_type):
	x = np.load(data_dir + "A0" + str(num) + file_type + "D_cropped.npy").astype(np.float32)
	y = np.load(data_dir + "A0" + str(num) + file_type + "K_cropped.npy").astype(np.float32)
	return x, y

model = EEGNet3D(4)

model.compile(loss='sparse_categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

model.summary()

X, Y = load_data(DATA_DIR, 1, "T")

model.fit(X, Y, epochs=10)

_, acc = model.evaluate(X, Y)

print("Accuracy: %.2f" % (acc*100))
