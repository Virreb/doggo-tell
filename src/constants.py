import multiprocessing as mp

# FOLDERS = ["barking", "growling", "howling", "whining"]
FOLDERS = ["barking", "growling", "whining", "howling", "other"]
FOLDER_ENCODING = {key: val for val, key in enumerate(FOLDERS)}
NBR_MFCC_FEATURES = 20
SAMPLING_RATE = 22050
MAX_FRAMES = 90
FEATURE_VECTOR_LENGTH = MAX_FRAMES * NBR_MFCC_FEATURES  # max len is 85 in data
NOT_FEATURE_COLUMNS = ["nbr_frames", "used_rate", "file_rate", "file_path", "target"]
NBR_WORKERS = mp.cpu_count() - 1
