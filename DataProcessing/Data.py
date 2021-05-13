# Data Processing Code 
# 	This is a followup attempt at the data processing code, it was becoming a bit unmanaged
#	and I wanted to rework it to be more dynamic and easier to use.

import os, mne, sys, re
from scipy.io import loadmat
import numpy as np
from glob import glob

FILE_PATTERN = "A0[1-9]"

DEFAULT_DIR = "../../datasets/BCICIV_2a_gdf/"

PROCESSED_DIR = "../../datasets/BCICIV_2a_processed/"

PROCESSED_2D_DIR = "../../datasets/BCICIV_2a_2d_processed/"

CROPPED_DIR = "../../datasets/BCICIV_2a_cropped/"

AVERAGED_DIR = "../../datasets/BCICIV_2a_averaged/"

SHUFFLED_DIR = "../../datasets/BCICIV_2a_shuffled/"

EVAL_KEYS_SUB_DIR = "true_labels/"

IGNORE = ('EOG-left', 'EOG-central', 'EOG-right')

SAVE_MEMORY = True

USE03BOUNDS = True

class DataProcessing:
	def __init__(self, data_dir=DEFAULT_DIR):
		self.evalData = {}
		self.process_eval = False
		self.process_training = False
		file_type = input("What type of file to process, valid types include 'T', 'E', or 'B': ")
		while (file_type != "T" and file_type != "E" and file_type != "B"):
			print(file_type)
			file_type = input("Invalid input, please enter 'T', 'E', or 'B': ")
		if (file_type == "B"):
			file_type = "ET"
		self.file_type = file_type
		if ("E" in file_type):
			eval_keys_pattern = data_dir + EVAL_KEYS_SUB_DIR + FILE_PATTERN + "E.mat" 
			self.eval_keys = glob(eval_keys_pattern)
			glob_pattern_eval = data_dir + FILE_PATTERN + 'E.gdf'
			self.eval_data_files = glob(glob_pattern_eval)
			self.process_eval = True
		if ("T" in file_type):
			glob_pattern_training = data_dir + FILE_PATTERN + "T.gdf"
			self.training_data_files = glob(glob_pattern_training)
			self.process_training = True
		if (self.process_training):
			print("Found {} training data files".format(len(self.training_data_files)))
		if (self.process_eval):
			if (len(self.eval_keys) != len(self.eval_data_files)):
				print("Number of evaluation data files does not match key files, please check files")
				self.process_eval = False
				print("Please reinitialize with proper files to process evaluation data")
			else:
				print("Found {} evaluation data files".format(len(self.eval_data_files)));
		self.ignore = IGNORE

	def set_events(self, event_codes):
		self.events = event_codes
		self.event_dict = {}
		for i, event in enumerate(event_codes):
			self.event_dict[event] = i

	def set_dimensions_3d(self, x, y, dur):
		self.xDim = x
		self.yDim = y
		self.duration = dur

	def set_dimensions_2d(self, ch, dur):
		self.channels = ch
		self.duration = dur

	def __getSlice1D(self, raw, channel, dur, index):
		return raw[channel, index:index+dur]

	def __getSliceFull3D(self, raw, dur, index):
		data = raw.get_data()
		trial = np.zeros((self.xDim, self.yDim, self.duration))
		for i, channel in enumerate(raw.ch_names):
			if not channel in self.ignore:
				x, y = self.__convertIndices(channel)
				trial[x, y] = self.__getSlice1D(data, i, dur, index)
		return trial

	def __getSliceFull2D(self, raw, dur, index):
		data= raw.get_data()
		trial = np.zeros((self.channels, self.duration))
		for i, channel in enumerate(raw.ch_names):
			if not channel in self.ignore:
				trial[i] = self.__getSlice1D(data, i, dur, index)
		return trial

	def __convertIndices(self, channel):
		xDict = {'EEG-Fz':3, 'EEG-0':1, 'EEG-1':2, 'EEG-2':3, 'EEG-3':4, 'EEG-4':5, 'EEG-5':0, 'EEG-C3':1, 'EEG-6':2, 'EEG-Cz':3, 'EEG-7':4, 'EEG-C4':5, 'EEG-8':6, 'EEG-9':1, 'EEG-10':2, 'EEG-11':3, 'EEG-12':4, 'EEG-13':5, 'EEG-14':2, 'EEG-Pz':3, 'EEG-15':4, 'EEG-16':3} 
		yDict = {'EEG-Fz':0, 'EEG-0':1, 'EEG-1':1, 'EEG-2':1, 'EEG-3':1, 'EEG-4':1, 'EEG-5':2, 'EEG-C3':2, 'EEG-6':2, 'EEG-Cz':2, 'EEG-7':2, 'EEG-C4':2, 'EEG-8':2, 'EEG-9':3, 'EEG-10':3, 'EEG-11':3, 'EEG-12':3, 'EEG-13':3, 'EEG-14':4, 'EEG-Pz':4, 'EEG-15':4, 'EEG-16':5} 
		return xDict[channel], yDict[channel]

	def processData3D(self):
		try: self.xDim, self.yDim, self.duration
		except NameError: 
			print("Data x-dim, y-dim, and/or duration undefined, please call set_dimensions(x, y, dur) before constructing data")
			return
		try: self.events
		except NameError:
			print("Data events undefined, please call set_events(event_codes) before constructing data")
			return
		self.processed_training_data = {}
		if ("T" in self.file_type):
			for f in self.training_data_files:
				subjectNum = int(re.findall(r'%s(\d+)' % 'A0', f)[0])
				event_times = []
				event_types = []
				raw = mne.io.read_raw_gdf(f, verbose='ERROR')
				for i in range(len(raw.annotations)):
					if (raw.annotations[i]['description'] in self.events):
						event_times.append(int(raw.annotations[i]['onset']*250))
						event_types.append(self.event_dict[raw.annotations[i]['description']])
				key = np.array(event_types)
				data = np.zeros((len(event_times), self.xDim, self.yDim, self.duration))
				for i, event in enumerate(event_times):
					sys.stdout.write("\rProcessing Training Trial {} from subject {}".format(i+1, subjectNum))				
					data[i] = self.__getSliceFull3D(raw, self.duration, event_times[i])
					sys.stdout.flush()
				sys.stdout.write("\rFinished processing training trials from subject {}\n".format(subjectNum))
				self.processed_training_data[subjectNum] = [data, key]
		self.processed_eval_data = {}
		if ("E" in self.file_type):
			for f in self.eval_data_files:
				subjectNum = int(re.findall(r'%s(\d+)' % 'A0', f)[0])
				event_times = []
				event_types = []
				raw = mne.io.read_raw_gdf(f, verbose='ERROR')
				for i in range(len(raw.annotations)):
					if (raw.annotations[i]['description'] == '768'):
						event_times.append(int(raw.annotations[i]['onset']*250))
				data = np.zeros((len(event_times), self.xDim, self.yDim, self.duration))
				for i, event in enumerate(event_times):
					sys.stdout.write("\rProcessing Evaluation Trial {} from subject {}".format(i+1, subjectNum))
					data[i] = self.__getSliceFull3D(raw, self.duration, event_times[i])
					sys.stdout.flush()
				sys.stdout.write("\rFinished processing evaluation trials from subject {}\n".format(subjectNum))
				self.processed_eval_data[subjectNum] = [data, None]
			for f in self.eval_keys:
				subjectNum = int(re.findall(r'%s(\d+)' % 'A0', f)[0])
				raw = loadmat(f)
				self.processed_eval_data[subjectNum][1] = raw['classlabel']
				if (USE03BOUNDS):
					self.processed_eval_data[subjectNum][1] -= 1

	def processData2D(self):
		try: self.channels, self.duration
		except NameError: 
			print("Data channels and/or duration undefined, please call set_dimensions_2d(channels dur) before constructing data")
			return
		try: self.events
		except NameError:
			print("Data events undefined, please call set_events(event_codes) before constructing data")
			return
		self.processed_training_data_2d = {}
		if ("T" in self.file_type):
			for f in self.training_data_files:
				subjectNum = int(re.findall(r'%s(\d+)' % 'A0', f)[0])
				event_times = []
				event_types = []
				raw = mne.io.read_raw_gdf(f, verbose='ERROR')
				for i in range(len(raw.annotations)):
					if (raw.annotations[i]['description'] in self.events):
						event_times.append(int(raw.annotations[i]['onset']*250))
						event_types.append(self.event_dict[raw.annotations[i]['description']])
				key = np.array(event_types)
				data = np.zeros((len(event_times), self.channels, self.duration))
				for i, event in enumerate(event_times):
					sys.stdout.write("\rProcessing Training Trial {} from subject {}".format(i+1, subjectNum))				
					data[i] = self.__getSliceFull2D(raw, self.duration, event_times[i])
					sys.stdout.flush()
				sys.stdout.write("\rFinished processing training trials from subject {}\n".format(subjectNum))
				self.processed_training_data_2d[subjectNum] = [data, key]
		self.processed_eval_data_2d = {}
		if ("E" in self.file_type):
			for f in self.eval_data_files:
				subjectNum = int(re.findall(r'%s(\d+)' % 'A0', f)[0])
				event_times = []
				event_types = []
				raw = mne.io.read_raw_gdf(f, verbose='ERROR')
				for i in range(len(raw.annotations)):
					if (raw.annotations[i]['description'] == '768'):
						event_times.append(int(raw.annotations[i]['onset']*250))
				data = np.zeros((len(event_times), self.channels, self.duration))
				for i, event in enumerate(event_times):
					sys.stdout.write("\rProcessing Evaluation Trial {} from subject {}".format(i+1, subjectNum))
					data[i] = self.__getSliceFull2D(raw, self.duration, event_times[i])
					sys.stdout.flush()
				sys.stdout.write("\rFinished processing evaluation trials from subject {}\n".format(subjectNum))
				self.processed_eval_data_2d[subjectNum] = [data, None]
			for f in self.eval_keys:
				subjectNum = int(re.findall(r'%s(\d+)' % 'A0', f)[0])
				raw = loadmat(f)
				self.processed_eval_data_2d[subjectNum][1] = raw['classlabel']
				if (USE03BOUNDS):
					self.processed_eval_data_2d[subjectNum][1] -= 1

	def saveProcessedData(self):
		try: self.processed_training_data, self.processed_eval_data
		except (NameError, AttributeError):
			print("Data not yet processed, please call processData() before this.")
			return
		if ("T" in self.file_type):
			for key in self.processed_training_data:
				fileBase = PROCESSED_DIR + "A0" + str(key) + "T"
				np.save(fileBase + "D_processed.npy", self.processed_training_data[key][0])
				np.save(fileBase + "K_processed.npy", self.processed_training_data[key][1])
		if ("E" in self.file_type):
			for key in self.processed_eval_data:
				fileBase = PROCESSED_DIR + "A0" + str(key) + "E"
				np.save(fileBase + "D_processed.npy", self.processed_eval_data[key][0])
				np.save(fileBase + "K_processed.npy", self.processed_eval_data[key][1])

	def saveProcessedData2D(self):
		try: self.processed_training_data_2d, self.processed_eval_data_2d
		except (NameError, AttributeError):
			print("Data not yet processed, please call processData2D() before this.")
			return
		if ("T" in self.file_type):
			for key in self.processed_training_data_2d:
				fileBase = PROCESSED_2D_DIR + "A0" + str(key) + "T"
				np.save(fileBase + "D_processed.npy", self.processed_training_data_2d[key][0])
				np.save(fileBase + "K_processed.npy", self.processed_training_data_2d[key][1])
		if ("E" in self.file_type):
			for key in self.processed_eval_data_2d:
				fileBase = PROCESSED_2D_DIR + "A0" + str(key) + "E"
				np.save(fileBase + "D_processed.npy", self.processed_eval_data_2d[key][0])
				np.save(fileBase + "K_processed.npy", self.processed_eval_data_2d[key][1])

	def loadProcessedData(self, file_type="B"):
		if (file_type != "T" and file_type != "E" and file_type != "B"):
			print("Please call loadProcessedData(file_type) with a value file_type of either 'T', 'E', or 'B'")
			return
		self.processed_training_data = {}
		self.processed_eval_data = {}
		if ("B" == file_type):
			file_type = "TE";
		self.file_type = file_type
		if ("T" in file_type):
			try:
				fileBase = PROCESSED_DIR + "A0"
				fileSuffD = "TD_processed.npy"
				fileSuffK = "TK_processed.npy"
				for key in range(1, 10):
					d = np.load(fileBase + str(key) + fileSuffD)
					k = np.load(fileBase + str(key) + fileSuffK)
					self.processed_training_data[key] = [d, k]
			except FileNotFoundError:
				self.processed_training_data = {}
				print("Processed Training Data not found, please check file system")
		if ("E" in file_type):
			try:
				fileBase = PROCESSED_DIR + "A0"
				fileSuffD = "ED_processed.npy"
				fileSuffK = "EK_processed.npy"
				for key in range(1, 10):
					d = np.load(fileBase + str(key) + fileSuffD)
					k = np.load(fileBase + str(key) + fileSuffK)
					self.processed_eval_data[key] = [d, k]
			except FileNotFoundError:
				self.processed_eval_data = {}
				print("Processed Evaluation Data not found, please check file system")

	def saveSingleCroppedSubject(self, subjectNum, file_type):
		try:
			if ("T" in file_type):
				self.cropped_training_data
			if ("E" in file_type):
				self.cropped_eval_data
		except (NameError, AttributeError):
			print("Data not yet processed, please call processData() or loadProcessedData() before calling saveSingleCroppedSubject()")
			return
		fileBase = CROPPED_DIR + "A0" + str(subjectNum) + file_type
		if (file_type == "T"):
			np.save(fileBase + "D_cropped.npy", self.cropped_training_data[subjectNum][0])
			np.save(fileBase + "K_cropped.npy", self.cropped_training_data[subjectNum][1])
		if (file_type == "E"):
			np.save(fileBase + "D_cropped.npy", self.cropped_eval_data[subjectNum][0])
			np.save(fileBase + "K_cropped.npy", self.cropped_eval_data[subjectNum][1])
		

	def cropData(self, window_size=240, save_mem=SAVE_MEMORY):
		try: self.processed_training_data, self.processed_eval_data
		except (NameError, AttributeError):
			print("Data not yet processed, please call processData() or loadProcessedData() before calling cropData()")
			return
		steps = self.duration - window_size + 1
		self.window_size = window_size
		self.cropped_training_data = {}
		if ("T" in self.file_type):
			numSubjects = len(self.processed_training_data.keys())
			for key in range(1, numSubjects+1):
				l = []
				for i in range(len(self.processed_training_data[key][0])):
					for j in range(steps):
						l.append(self.processed_training_data[key][0][i, :, :, j:j+window_size])
				d = np.array(l)
				l = []
				for i in range(len(self.processed_training_data[key][1])):
					for j in range(steps):
						l.append(self.processed_training_data[key][1][i])
				k = np.array(l)
				self.cropped_training_data[key] = [d, k]
				del(self.processed_training_data[key])
				if (save_mem):
					self.saveSingleCroppedSubject(key, "T")
					del(self.cropped_training_data[key])
		self.cropped_eval_data = {}
		if ("E" in self.file_type):
			for key in range(1, numSubjects+1):
				l = []
				for i in range(len(self.processed_eval_data[key][0])):
					for j in range(steps):
						l.append(self.processed_eval_data[key][0][i, :, :, j:j+window_size])
				d = np.array(l)
				l = []
				for i in range(len(self.processed_eval_data[key][1])):
					for j in range(steps):
						l.append(self.processed_eval_data[key][1][i])
				k = np.array(l)
				self.cropped_eval_data[key] = [d, k]
				del(self.processed_eval_data[key])
				if (save_mem):
					self.saveSingleCroppedSubject(key, "E")
					del(self.cropped_eval_data[key])

	def saveCroppedData(self):
		try: 
			if ("T" in self.file_type):
				self.cropped_training_data
			if ("E" in self.file_type):
				self.cropped_eval_data
		except NameError:
			print("Data not yet cropped, please call cropData() or loadCroppedData() before this.")
			return
		if ("T" in self.file_type):
			for key in self.cropped_training_data:
				fileBase = CROPPED_DIR + "A0" + str(key) + "T"
				np.save(fileBase + "D_cropped.npy", self.cropped_training_data[key][0])
				np.save(fileBase + "K_cropped.npy", self.cropped_training_data[key][1])
		if ("E" in self.file_type):
			for key in self.cropped_eval_data:
				fileBase = CROPPED_DIR + "A0" + str(key) + "E"
				np.save(fileBase + "D_cropped.npy", self.cropped_eval_data[key][0])
				np.save(fileBase + "K_cropped.npy", self.cropped_eval_data[key][1])

	def loadCroppedData(self, file_type="B"):
		if (file_type != "T" and file_type != "E" and file_type != "B"):
			print("Please call loadCroppedData(file_type) with a value file_type of either 'T', 'E', or 'B'")
			return
		self.cropped_training_data = {}
		self.cropped_eval_data = {}
		if ("B" == file_type):
			file_type = "TE";
		self.file_type = file_type
		if ("T" in file_type):
			try:
				fileBase = CROPPED_DIR + "A0"
				fileSuffD = "TD_cropped.npy"
				fileSuffK = "TK_cropped.npy"
				for key in range(1, 10):
					d = np.load(fileBase + str(key) + fileSuffD)
					k = np.load(fileBase + str(key) + fileSuffK)
					self.cropped_training_data[key] = [d, k]
			except FileNotFoundError:
				self.cropped_training_data = {}
				print("Cropped Training Data not found, please check file system")
		if ("E" in file_type):
			try:
				fileBase = CROPPED_DIR + "A0"
				fileSuffD = "ED_cropped.npy"
				fileSuffK = "EK_cropped.npy"
				for key in range(1, 10):
					d = np.load(fileBase + str(key) + fileSuffD)
					k = np.load(fileBase + str(key) + fileSuffK)
					self.cropped_eval_data[key] = [d, k]
			except FileNotFoundError:
				self.cropped_eval_data = {}
				print("Cropped Evaluation Data not found, please check file system")

	def averageData(self):
		try: 
			if ("T" in self.file_type):
				self.cropped_training_data
			if ("E" in self.file_type):
				self.cropped_eval_data
		except NameError:
			print("Data not yet cropped, please call cropData() or loadCroppedData() before this.")
			return
		if ("T" in self.file_type):
			self.averaged_training_data = {}
			for key in self.cropped_training_data:
				self.averaged_training_data[key] = self.cropped_training_data[key]
				for sample in range(len(self.cropped_training_data[key][0])):
					a = np.zeros((7, 6))
					for i in range(7):
						for j in range(6):
							a[i, j] = np.average(self.cropped_training_data[key][0][sample][i, j])
					a = np.repeat(a[:, :, np.newaxis], 240, axis=2)
					self.averaged_training_data[key][0][sample] = self.cropped_training_data[key][0][sample] - a	
		if ("E" in self.file_type):
			self.averaged_eval_data = {}
			for key in self.cropped_eval_data:
				self.averaged_eval_data[key] = self.cropped_eval_data[key]
				for sample in range(len(self.cropped_eval_data[key][0])):
					a = np.zeros((7, 6))
					for i in range(7):
						for j in range(6):
							a[i, j] = np.average(self.cropped_eval_data[key][0][sample][i, j])
					a = np.repeat(a[:, :, np.newaxis], 240, axis=2)
					self.averaged_eval_data[key][0][sample] = self.cropped_eval_data[key][0][sample] - a	


	def saveAveragedData(self):
		try: 
			if ("T" in self.file_type):
				self.averaged_training_data
			if ("E" in self.file_type):
				self.averaged_eval_data
		except NameError:
			print("Data not yet averaged, please call averageData() or loadAveragedData() before this.")
			return
		if ("T" in self.file_type):
			for key in self.averaged_training_data:
				fileBase = AVERAGED_DIR + "A0" + str(key) + "T"
				np.save(fileBase + "D_averaged.npy", self.averaged_training_data[key][0])
				np.save(fileBase + "K_averaged.npy", self.averaged_training_data[key][1])
		if ("E" in self.file_type):
			for key in self.averaged_eval_data:
				fileBase = AVERAGED_DIR + "A0" + str(key) + "E"
				np.save(fileBase + "D_averaged.npy", self.averaged_eval_data[key][0])
				np.save(fileBase + "K_averaged.npy", self.averaged_eval_data[key][1])

	def loadAveragedData(self, file_type="B"):
		if (file_type != "T" and file_type != "E" and file_type != "B"):
			print("Please call loadAveragedData(file_type) with a value file_type of either 'T', 'E', or 'B'")
			return
		self.averaged_training_data = {}
		self.averaged_eval_data = {}
		if ("B" == file_type):
			file_type = "TE";
		self.file_type = file_type
		if ("T" in file_type):
			try:
				fileBase = AVERAGED_DIR + "A0"
				fileSuffD = "TD_averaged.npy"
				fileSuffK = "TK_averaged.npy"
				for key in range(1, 10):
					d = np.load(fileBase + str(key) + fileSuffD)
					k = np.load(fileBase + str(key) + fileSuffK)
					self.averaged_training_data[key] = [d, k]
			except FileNotFoundError:
				self.averaged_training_data = {}
				print("Averaged Training Data not found, please check file system")
		if ("E" in file_type):
			try:
				fileBase = AVERAGED_DIR + "A0"
				fileSuffD = "ED_averaged.npy"
				fileSuffK = "EK_averaged.npy"
				for key in range(1, 10):
					d = np.load(fileBase + str(key) + fileSuffD)
					k = np.load(fileBase + str(key) + fileSuffK)
					self.averaged_eval_data[key] = [d, k]
			except FileNotFoundError:
				self.averaged_eval_data = {}
				print("Averaged Evaluation Data not found, please check file system")

	def shuffleData(self):
		try: 
			if ("T" in self.file_type):
				self.averaged_training_data
			if ("E" in self.file_type):
				self.averaged_eval_data
		except NameError:
			print("Data not yet averaged, please call averageData() or loadAveragedData() before this.")
			return
		if ("T" in self.file_type):
			self.shuffled_training_data = {}
			for key in self.averaged_training_data:
				self.shuffled_training_data[key] = self.averaged_training_data[key]
				p = np.random.permutation(len(self.averaged_training_data[key][0]))
				self.shuffled_training_data[key][0] = self.averaged_training_data[key][0][p]
				self.shuffled_training_data[key][1] = self.averaged_training_data[key][1][p]
		if ("E" in self.file_type):
			self.shuffled_eval_data = {}
			for key in self.averaged_eval_data:
				self.shuffled_eval_data[key] = self.averaged_eval_data[key]
				p = np.random.permutation(len(self.averaged_eval_data[key][0]))
				self.shuffled_eval_data[key][0] = self.averaged_eval_data[key][0][p]
				self.shuffled_eval_data[key][1] = self.averaged_eval_data[key][1][p]
			
	def saveShuffledData(self):
		try: 
			if ("T" in self.file_type):
				self.shuffled_training_data
			if ("E" in self.file_type):
				self.shuffled_eval_data
		except NameError:
			print("Data not yet averaged, please call shuffleData() or loadShuffledData() before this.")
			return
		if ("T" in self.file_type):
			for key in self.shuffled_training_data:
				fileBase = SHUFFLED_DIR + "A0" + str(key) + "T"
				np.save(fileBase + "D_shuffled.npy", self.shuffled_training_data[key][0])
				np.save(fileBase + "K_shuffled.npy", self.shuffled_training_data[key][1])
		if ("E" in self.file_type):
			for key in self.shuffled_eval_data:
				fileBase = SHUFFLED_DIR + "A0" + str(key) + "E"
				np.save(fileBase + "D_shuffled.npy", self.shuffled_eval_data[key][0])
				np.save(fileBase + "K_shuffled.npy", self.shuffled_eval_data[key][1])

	def loadShuffledData(self):
		if (file_type != "T" and file_type != "E" and file_type != "B"):
			print("Please call loadShuffledData(file_type) with a value file_type of either 'T', 'E', or 'B'")
			return
		self.shuffled_training_data = {}
		self.shuffled_eval_data = {}
		if ("B" == file_type):
			file_type = "TE";
		self.file_type = file_type
		if ("T" in file_type):
			try:
				fileBase = SHUFFLED_DIR + "A0"
				fileSuffD = "TD_shuffled.npy"
				fileSuffK = "TK_shuffled.npy"
				for key in range(1, 10):
					d = np.load(fileBase + str(key) + fileSuffD)
					k = np.load(fileBase + str(key) + fileSuffK)
					self.shuffled_training_data[key] = [d, k]
			except FileNotFoundError:
				self.shuffled_training_data = {}
				print("Shuffled Training Data not found, please check file system")
		if ("E" in file_type):
			try:
				fileBase = SHUFFLED_DIR + "A0"
				fileSuffD = "ED_shuffled.npy"
				fileSuffK = "EK_shuffled.npy"
				for key in range(1, 10):
					d = np.load(fileBase + str(key) + fileSuffD)
					k = np.load(fileBase + str(key) + fileSuffK)
					self.shuffled_eval_data[key] = [d, k]
			except FileNotFoundError:
				self.shuffled_eval_data = {}
				print("Shuffled Evaluation Data not found, please check file system")
