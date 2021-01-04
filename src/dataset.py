import os
import librosa
import numpy as np
import nlpaug.augmenter.audio as naa
from tqdm import tqdm
from shutil import copyfile
import pandas as pd
from src.constants import FOLDERS, NBR_MFCC_FEATURES, SAMPLING_RATE, FOLDER_ENCODING

max_len = 90 * NBR_MFCC_FEATURES  # max len is 85 in data


def train_test_split(test_ratio=0.2):
    print("\nSplitting data to train and test")
    for folder in FOLDERS:
        print(f"\t'{folder}' to {test_ratio} test and {1-test_ratio} train")

        for new_folder_path in [f"data/test/{folder}", f"data/train/{folder}"]:
            if os.path.exists(new_folder_path) is False:
                os.makedirs(new_folder_path)

        raw_folder_path = f"data/raw/{folder}"
        for file_name in os.listdir(raw_folder_path):
            current_file_path = f"{raw_folder_path}/{file_name}"

            if np.random.rand() <= test_ratio:
                new_file_path = f"data/test/{folder}/{file_name}"
            else:
                new_file_path = f"data/train/{folder}/{file_name}"

            copyfile(current_file_path, new_file_path)


def create_features(data_slice="train", augment_loudness=True):
    all_features = []
    labels = {key: 0 for key in FOLDERS}
    aug = naa.LoudnessAug(factor=(2, 5))  # TODO: Check these factors

    for folder in FOLDERS:  # TODO: Parallize this
        folder_path = f"data/{data_slice}/{folder}"
        print(f"\nCreating {data_slice}ing features for '{folder}'")

        for file_name in tqdm(os.listdir(folder_path)):
            file_path = f"{folder_path}/{file_name}"

            # load the wav file
            file_data, file_rate = librosa.load(file_path)

            if augment_loudness:
                # loudness augmenter (where file_data is the output of librosa.load)
                file_data = aug.augment(file_data)      # TODO: Test should not be augmented

            # get mfcc in 2D
            mfcc = librosa.feature.mfcc(y=file_data, sr=SAMPLING_RATE, n_mfcc=NBR_MFCC_FEATURES)
            nbr_frames = mfcc.shape[1]

            # convert mfcc in 2D to mfcc in 1D
            mfcc = mfcc.flatten()     # TODO: could take the mean over time for every MFCC here instead

            # pad mfcc so that all files have features in the same length
            mfcc = np.pad(mfcc, (0, max_len - len(mfcc)), 'constant')

            # save to dict in order to create DF later
            features = {f"mfcc_{k}": val for k, val in enumerate(mfcc)}

            # TODO: add more features from librosa?

            new_labels = labels.copy()
            new_labels[folder] = 1
            # features = {**features, **new_labels}
            features["target"] = FOLDER_ENCODING[folder]
            features["file_path"] = file_path
            features["file_rate"] = file_rate
            features["used_rate"] = SAMPLING_RATE
            features["nbr_frames"] = nbr_frames

            all_features.append(features)

    df = pd.DataFrame(all_features)
    df.to_pickle(f"data/{data_slice}/df.pkl")

    return df
