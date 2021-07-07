from MB3DEEGNet import EEGNet3D
from tensorflow.keras import utils as np_utils
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import sys
import os

print("Tensorflow Executing Eagerly: {}".format(tf.executing_eagerly()))

THREADING_CONTROL = False
NUM_THREADS = 2

PROFILING = True
PROFILING_OUT_DIR = "./tb_out"

NUM_SUBJECTS = 9
UNTRAINED_DATA_RATIO = 0.1

KFOLD = True
KFOLD_NUM = 5

TYPE = "UNTRAINED"

UNTRAINED_TYPE = "OUTSIDE"

SUBJECT = sys.argv[1]

OUT_FILE = "./results"

DATA_DIR = "/share/multibranch/datasets/BCICIV_2a_shuffled/"

BRANCHES = ("SRF", "MRF", "LRF")
N = 4

class CustomCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		print("Finished epoch {} with Loss: {} and Accuracy: {}".format(epoch+1, logs['loss'], logs['accuracy']))
		sys.stdout.flush()

CALLBACKS = [CustomCallback()]
if (PROFILING):
	tb_callback = tf.keras.callbacks.TensorBoard(log_dir=PROFILING_OUT_DIR, profile_batch=2)
	CALLBACKS.append(tb_callback)

if (THREADING_CONTROL):
	tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
	tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
	tf.config.set_soft_device_placement(True)

physical_devices = tf.config.list_physical_devices('GPU')
try:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
	print("Invalid device or cannot modify virtual device once initialized")
	pass

def load_data(data_dir, num, et):
	x = np.load(data_dir + "A0" + str(num) + str(et) + "D_shuffled.npy", encoding='latin1', allow_pickle=True).astype(np.float64)
	y = np.load(data_dir + "A0" + str(num) + str(et) + "K_shuffled.npy", encoding='latin1', allow_pickle=True).astype(np.float64)
	return x, y

if (THREADING_CONTROL):
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

trialNum = int(sys.argv[1])

init_lr = 0.0001

print("Verifying thread usage params")
print("Intra Op Threads:\t", tf.config.threading.get_intra_op_parallelism_threads())
print("Inter Op Threads:\t", tf.config.threading.get_inter_op_parallelism_threads())

sys.stdout.flush()

loss_functions = ('sparse_categorical_crossentropy', 'mean_absolute_error', 'mean_squared_error', 'categorical_crossentropy')

optimizers = [tf.keras.optimizers.Adam(init_lr), tf.keras.optimizers.SGD(init_lr)]

x_subs = []
y_subs = []

if (TYPE == "TRAINED"):
	x_tmp, y_tmp = load_data(DATA_DIR, int(SUBJECT), "T")
	x_subs.append(x_tmp)
	y_subs.append(y_tmp)

elif (TYPE == "UNTRAINED"):
	for subject in range(1, NUM_SUBJECTS+1):
		if subject == int(SUBJECT):
			continue
		x_tmp, y_tmp = load_data(DATA_DIR, int(subject), "T")
		x_subs.append(x_tmp[0:int(UNTRAINED_DATA_RATIO*x_tmp.shape[0])])
		y_subs.append(y_tmp[0:int(UNTRAINED_DATA_RATIO*y_tmp.shape[0])])

	x_subject, y_subject = load_data(DATA_DIR, int(SUBJECT), "T")

X_data = np.concatenate(x_subs)
Y_data = np.concatenate(y_subs)

p = np.random.permutation(len(X_data))
X_data = X_data[p]
Y_data = Y_data[p]

print("Data Loaded")
sys.stdout.flush()

if (KFOLD):
	kfold = KFold(n_splits=KFOLD_NUM, shuffle=True)

	acc_per_fold = []
	loss_per_fold = []

	acc_arr = []
	val_acc_arr = []
	loss_arr = []
	val_loss_arr = []


	fold_no = 1

	print("Beginning K-Fold Analysis")
	sys.stdout.flush()
	for train, test in kfold.split(X_data, Y_data):
		print("Performing K-Fold {}".format(fold_no), flush=True)

		if (TYPE == "TRAINED"):
			x_train, x_val, y_train, y_val = train_test_split(X_data[train], Y_data[train], test_size=0.1)
			x_test, y_test = X_data[test], Y_data[test]

		elif (TYPE == "UNTRAINED"):
			x_train, x_trained_test, y_train, y_trained_test = train_test_split(X_data[train], Y_data[train], test_size=0.1)
			x_val, x_test, y_val, y_test = train_test_split(x_subject, y_subject, test_size=0.5)

		print("Getting Model", flush=True)
		model = EEGNet3D(N, kernLength = 125, branches = BRANCHES)
		model.compile(loss=loss_functions[0], optimizer=optimizers[0], metrics=['accuracy'])
		print("Beginning Model Training", flush=True)
		history = model.fit(x_train, y_train, epochs=300, batch_size=50, validation_data=(x_val, y_val), verbose=0, callbacks=CALLBACKS)
		print("Model Trained", flush=True)	
	
		acc_arr.append(history.history['accuracy'])
		val_acc_arr.append(history.history['val_accuracy'])
		loss_arr.append(history.history['loss'])
		val_loss_arr.append(history.history['val_loss'])
			
		print("Evaluating Model Performance", flush=True)
		if (TYPE != "UNTRAINED" or UNTRAINED_TYPE == "OUTSIDE"):
			loss, acc = model.evaluate(x_test, y_test, verbose=0, callbacks=CALLBACKS)

		else:
			loss, acc = model.evaluate(x_trained_test, y_trained_test, verbose=0, callbacks=CALLBACKS)

		sys.stdout.flush()
		print("Accuracy: %.2f" % acc*100)
		sys.stdout.flush()

		acc_per_fold.append(acc)
		loss_per_fold.append(loss)
		fold_no += 1

	fig, ax = plt.subplots(nrows=KFOLD_NUM+1, ncols=2, figsize=(8, 2*(KFOLD_NUM+1)))

	for k in range(KFOLD_NUM):
		ax[k, 0].plot(acc_arr[k], color='blue', label="K-Fold {} train acc".format(k+1))
		ax[k, 0].plot(val_acc_arr[k], color='orange', label="K-Fold {} test acc".format(k+1))

		ax[k, 1].plot(loss_arr[k], color='orange', label="K-Fold {} train loss".format(k+1))
		ax[k, 1].plot(val_loss_arr[k], color='blue', label="K-Fold {} test loss".format(k+1))

	total_acc = np.mean(acc_arr, axis=0)
	total_val_acc = np.mean(val_acc_arr, axis=0)
	total_loss = np.mean(loss_arr, axis=0)
	total_val_loss = np.mean(val_loss_arr, axis=0)

	acc = np.mean(acc_per_fold)
	std = np.std(acc_per_fold)

	ax[KFOLD_NUM, 0].plot(total_acc, color='blue', label="Total Train Acc")
	ax[KFOLD_NUM, 0].plot(total_val_acc, color='orange', label="Total Test Acc")

	ax[KFOLD_NUM, 1].plot(total_loss, color='orange', label="Total Train Loss")
	ax[KFOLD_NUM, 1].plot(total_val_loss, color='blue', label="Total Test Loss")

	for axis in ax:
		axis[0].set_ylabel("Accuracy")
		axis[0].set_title("Accuracy eval")

		axis[1].set_ylabel("Loss")
		axis[1].set_xlabel("Epoch")
		axis[1].set_title("Loss eval")

	fig.suptitle("Job ID: {}; Acc: {:.4f} (+/- {:.4f})".format(sys.argv[2], acc, std))

	plt.tight_layout()

	fig.savefig("./seed_plt_figs/fig-{}-{}-{}.png".format(trialNum, KFOLD_NUM, sys.argv[2]), dpi=256)
	plt.close(fig)

with open(OUT_FILE, "a") as f:
	new_entry = "{} - Motor - {} - {} - {} - Subject {} - {}-Fold - {:.2f}%\n".format(sys.argv[2], TYPE, UNTRAINED_TYPE, BRANCHES, SUBJECT, KFOLD_NUM, acc*100)
	f.write(new_entry)

print("Accuracy: %.2f" % (acc*100))
