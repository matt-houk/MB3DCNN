import glob, os
import numpy as np
import mne

def getIndex(raw, tagIndex);
def isEvent(raw, tagIndex, event);
def getSlice1D(raw, channel, dur, index);
def getSliceFull(raw, dur, index);
def convertIndices(channel);

DURATION = 313

data_files = glob.glob('./BCICIV_2a_gdf/*.gdf')

print(data_files)

event_times = [];

raw = mne.io.read_raw_gdf(data_files[0], verbose='ERROR')

for i in range(len(raw.annotations)):
	if (isEvent(raw, i, '783')):
		event_times.append(getIndex(raw, i))

print(raw[0][0][0][int(t1):int(t1+t2)])

def getIndex(raw, tagIndex):
	return int(raw.annotations[tagIndex]['onset']*250)

def isEvent(raw, tagIndex, event):
	if (raw.annotations[tagIndex]['description'] == event):
		return True
	return False

def getSlice1D(raw, channel, dur, index):
	if (type(channel) == int):
		channel = raw.ch_names[channel]
	return raw[channel][0][0][index:index+dur]	

def getSliceFull(raw, dur, index):
	
def convertIndices(channel):
	xDict = {'EEG-0':3, 'EEG-4':3, 'EEG-10':3, 'EEG-16':3, 'EEG-20':3, 'EEG-22':3}
	yDict = {}

