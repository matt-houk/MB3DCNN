from Data import DataProcessing

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("Initializing Data Processing Object")

processor = DataProcessing()

print("Setting events")

processor.set_events(('769', '770', '771', '772'))

print("Setting dimensions of processed data")

processor.set_dimensions(7, 6, 313)

print("Loading processed data")

processor.loadProcessedData()

print("Cropping Data")

processor.cropData()

#print("Saving cropped data")

#processor.saveCroppedData()
