import glob, os
import numpy as np
import mne

SUBJECT = 1
DURATION = 313
XDIM = 7
YDIM = 6
IGNORE = ('EOG-left', 'EOG-central', 'EOG-right')

def getIndex(raw, tagIndex):
	return int(raw.annotations[tagIndex]['onset']*250)

def isEvent(raw, tagIndex, events):
	for event in events:
		if (raw.annotations[tagIndex]['description'] == event):
			return True
	return False

def getSlice1D(raw, channel, dur, index):
	return raw[channel][index:index+dur]	

def getSliceFull(raw, dur, index):
	data = raw.get_data()
	trial = np.zeros((XDIM, YDIM, dur))
	for i, channel in enumerate(raw.ch_names):
		if not channel in IGNORE:
			x, y = convertIndices(channel)
			trial[x][y] = getSlice1D(data, i, dur, index)
	return trial
	
def convertIndices(channel):
	xDict = {'EEG-Fz':3, 'EEG-0':1, 'EEG-1':2, 'EEG-2':3, 'EEG-3':4, 'EEG-4':5, 'EEG-5':0, 'EEG-C3':1, 'EEG-6':2, 'EEG-Cz':3, 'EEG-7':4, 'EEG-C4':5, 'EEG-8':6, 'EEG-9':1, 'EEG-10':2, 'EEG-11':3, 'EEG-12':4, 'EEG-13':5, 'EEG-14':2, 'EEG-Pz':3, 'EEG-15':4, 'EEG-16':3}
	yDict = {'EEG-Fz':0, 'EEG-0':1, 'EEG-1':1, 'EEG-2':1, 'EEG-3':1, 'EEG-4':1, 'EEG-5':2, 'EEG-C3':2, 'EEG-6':2, 'EEG-Cz':2, 'EEG-7':2, 'EEG-C4':2, 'EEG-8':2, 'EEG-9':3, 'EEG-10':3, 'EEG-11':3, 'EEG-12':3, 'EEG-13':3, 'EEG-14':4, 'EEG-Pz':4, 'EEG-15':4, 'EEG-16':5}
	return xDict[channel], yDict[channel]

data_files = glob.glob('../datasets/BCICIV_2a_gdf/*T.gdf')

try:
	raw = mne.io.read_raw_gdf(data_files[SUBJECT], verbose='ERROR')
except IndexError:
	print("No data files found") 

event_times = []
event_types = []

event_dict = {'769':1, '770':2, '771':3, '772':4}

for i in range(len(raw.annotations)):
	if (isEvent(raw, i, ('769', '770', '771', '772'))):
		event_times.append(getIndex(raw, i))
		event_types.append(event_dict[raw.annotations[i]['description']])

key = np.array(event_types)

data = np.zeros((len(event_times), XDIM, YDIM, DURATION))

for i, event in enumerate(event_times):
	print("Processing the " + str(i) + "th trial")
	data[i] = getSliceFull(raw, DURATION, event_times[i])

fnBase = raw.filenames[0][(raw.filenames[0].find("/A0")+1):raw.filenames[0].find(".gdf")]

fn0 = "../datasets/BCICIV_2a_processed/" + fnBase + ".npy"
fn1 = "../datasets/BCICIV_2a_processed/" + fnBase + "Key.npy"

np.save(fn1, key)
np.save(fn0, data)
