from Data import DataProcessing

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("Initializing Data Processing Object")

processor = DataProcessing()

print("Setting events")

processor.set_events(('769', '770', '771', '772'))

print("Setting dimensions of processed data")

processor.set_dimensions(7, 6, 313)

print("Loading cropped data")

processor.loadCroppedData(file_type="E")

print("Average cropped data")

processor.averageCroppedData()

print("Saving Averaged Data as cropped data")

processor.saveCroppedData()
