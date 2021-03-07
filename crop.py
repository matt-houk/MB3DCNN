import glob
import numpy as np

dir_pf = "../datasets/BCICIV_2a_processed/"
T_suff = "_cropped_shuffled.npy"
K_suff = "K_cropped_shuffled.npy"

df = glob.glob("../datasets/BCICIV_2a_processed/*.npy")

def Filter(string, substr):
	return [s for s in string if any(sub not in s for sub in substr)]

def Remove(string, substr):
	return [s.replace(substr, '') for s in string]

substr = ["cropped"]

df = Filter(df, substr)

substr = ["Key"]

df = Filter(df, substr)

df = Remove(df, '.npy')

print(df)
# Taking a numpy array that is [288][7][6][313] -> [288*74][7][6][240], should this be 73 or 74

for k in range(len(df)):
	n = np.load(df[k]+'.npy')
	l = []


	for i in range(0, 288):
		for j in range(0, 74):
			l.append(n[i, :, :, j:240+j])

	for r in l:
		a = np.zeros((7, 6))
		for i in range(7):
			for j in range(6):
				a[i, j] = np.average(r[i, j])
		a = np.repeat(a[:, :, np.newaxis], 240, axis=2)
		r = r - a;

	arr = np.array(l)
	np.random.seed(k)
	np.random.shuffle(arr)

	np.save(df[k] + T_suff, arr)

for k in range(len(df)):
	n = np.load(df[k]+'Key.npy')
	l = [] 
	for i in range(len(n)):
		for j in range(0, 74):
			l.append(n[i])

	arr = np.array(l)
	np.random.seed(k)
	np.random.shuffle(arr)

	np.save(df[k] + K_suff, arr)
