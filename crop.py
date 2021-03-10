# Imports
import glob
import numpy as np

# Directory and file suffixes - T is training and K is key
dir_pf = "../datasets/BCICIV_2a_processed/"
T_suff = "_cropped_shuffled.npy"
K_suff = "K_cropped_shuffled.npy"

# List of datafiles - This is not set up well, it has a tendency to grab extra files
df = glob.glob("../datasets/BCICIV_2a_processed/*.npy")

# Filters out strings from an array that have a substr
def Filter(string, substr):
	return [s for s in string if any(sub not in s for sub in substr)]

# Removes a substr from a string
def Remove(string, substr):
	return [s.replace(substr, '') for s in string]

# Removes files that are not the ones that need to be cropped
substr = ["cropped"]

df = Filter(df, substr)

substr = ["Key"]

df = Filter(df, substr)

df = Remove(df, '.npy')

print(df)

# Taking a numpy array that is [288][7][6][313] -> [288*74][7][6][240]
# Sliding window crop
for k in range(len(df)):
	# Load the data
	n = np.load(df[k]+'.npy')
	l = []

	# Grab the slices from the data array
	for i in range(0, 288):
		for j in range(0, 74):
			l.append(n[i, :, :, j:240+j])

	# Handle the normalization of each cropped array
	for r in l:
		a = np.zeros((7, 6))
		for i in range(7):
			for j in range(6):
				a[i, j] = np.average(r[i, j]) # Averaging each channel
		a = np.repeat(a[:, :, np.newaxis], 240, axis=2) # Turn the 2d array in a 3d array that repeats the same 2d array 240 times
		# Subtract the averaged array from each cropped array
		r = r - a;

	# Convert the list to an array
	arr = np.array(l)
	# Seed numpy and shuffle the array
	np.random.seed(k)
	np.random.shuffle(arr)

	# Save the cropped data
	np.save(df[k] + T_suff, arr)

# Handling the key dataset, as each key value must be repeated 74 times and shuffled
for k in range(len(df)):
	# Load the data
	n = np.load(df[k]+'Key.npy')
	l = [] 

	# Create a new list that has each key repeated 74 times
	for i in range(len(n)):
		for j in range(0, 74):
			l.append(n[i])

	# Convert the list to an array
	arr = np.array(l)
	# Seed numpy and shuffle the array
	np.random.seed(k)
	np.random.shuffle(arr)

	# Save the cropped data
	np.save(df[k] + K_suff, arr)
