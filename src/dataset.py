import os
import librosa
import numpy as np
import nlpaug.augmenter.audio as naa
from tqdm import tqdm
from shutil import copyfile, rmtree
import pandas as pd
from src.constants import FOLDERS, NBR_MFCC_FEATURES, SAMPLING_RATE, FOLDER_ENCODING, MAX_RAW_SOUND_POINTS
import sounddevice as sd
from scipy.io.wavfile import write
import time
import matplotlib.pyplot as plt


def train_test_split(test_ratio=0.2):
    print("\nSplitting data to train and test")
    for folder in FOLDERS:
        print(f"\t'{folder}' to {test_ratio} test and {1-test_ratio} train")

        for new_folder_path in [f"data/test/{folder}", f"data/train/{folder}"]:
            if os.path.exists(new_folder_path):
                rmtree(new_folder_path)
            os.makedirs(new_folder_path)

        raw_folder_path = f"data/raw/{folder}"
        for file_name in os.listdir(raw_folder_path):
            current_file_path = f"{raw_folder_path}/{file_name}"

            if np.random.rand() <= test_ratio:
                new_file_path = f"data/test/{folder}/{file_name}"
            else:
                new_file_path = f"data/train/{folder}/{file_name}"

            copyfile(current_file_path, new_file_path)


def create_audio_features(rec):
    # get mfcc in 2D
    mfcc = librosa.feature.mfcc(y=rec, sr=SAMPLING_RATE, n_mfcc=NBR_MFCC_FEATURES)
    nbr_frames = mfcc.shape[1]
    # print("mfcc frames", nbr_frames, mfcc.shape)
    # tmp_df = pd.DataFrame(mfcc.T)
    # print(tmp_df.head(40))
    # print(tmp_df.describe())

    # convert mfcc in 2D to mfcc in 1D
    mfcc = mfcc.flatten()  # TODO: could take the mean over time for every MFCC here instead

    # pad mfcc so that all files have features in the same length
    # mfcc = np.pad(mfcc, (0, FEATURE_VECTOR_LENGTH - len(mfcc)), 'constant')  # pad in the end
    # if np.random.rand() >= 0.5:
    #     mfcc = np.pad(mfcc, (0, FEATURE_VECTOR_LENGTH - len(mfcc)), 'constant')     # pad in the end
    # else:
    #     mfcc = np.pad(mfcc, (FEATURE_VECTOR_LENGTH - len(mfcc), 0), 'constant')     # pad in the beginning

    # TODO: add more features from librosa?

    # save to dict in order to create DF later
    features = {f"mfcc_{k}": val for k, val in enumerate(mfcc)}
    features["nbr_frames"] = nbr_frames

    return features


def create_features(data_slice="train", augment_loudness=True):
    all_features = []
    aug = naa.LoudnessAug(factor=(2, 5))  # TODO: Check these factors

    for folder in FOLDERS:  # TODO: Parallize this
        folder_path = f"data/{data_slice}/{folder}"
        print(f"\nCreating {data_slice}ing features for '{folder}'")

        # for file_name in os.listdir(folder_path):
        for file_name in tqdm(os.listdir(folder_path)):
            file_path = f"{folder_path}/{file_name}"

            # load the wav file
            file_data, file_rate = librosa.load(file_path, mono=True)

            if augment_loudness:
                # loudness augmenter (where file_data is the output of librosa.load)
                file_data = aug.augment(file_data)      # TODO: Test should not be augmented

            nbr_sound_points = file_data.shape[0]
            if nbr_sound_points > MAX_RAW_SOUND_POINTS:
                if np.random.rand() <= 0.5:
                    file_data = file_data[:MAX_RAW_SOUND_POINTS]  # only take first
                else:
                    file_data = file_data[MAX_RAW_SOUND_POINTS:]  # only take last
            else:
                if np.random.rand() <= 0.5:
                    file_data = np.pad(file_data, (0, MAX_RAW_SOUND_POINTS - nbr_sound_points), 'constant')  # pad in the end
                else:
                    file_data = np.pad(file_data, (MAX_RAW_SOUND_POINTS - nbr_sound_points, 0), 'constant')  # pad in the beginning

        # print("record len", file_data.shape[0])
            features = create_audio_features(file_data)
            # print("\n")
            # plt.plot(file_data)
            # plt.show()
            features["target"] = FOLDER_ENCODING[folder]
            features["file_path"] = file_path
            features["file_rate"] = file_rate
            features["used_rate"] = SAMPLING_RATE
            # features["nbr_raw_points"] = file_data.shape[0]

        # features = {"len": file_data.shape[0]}
            all_features.append(features)

    df = pd.DataFrame(all_features)
    # print(df.info())
    # print(df.describe())
    # print(df["nbr_frames"].describe())
    # exit(0)
    df.to_pickle(f"data/{data_slice}/df.pkl")

    return df


def record_ambient_sound(nbr_recordings=500, filename_prefix="custom_1"):

    if os.path.exists("data/other") is False:
        os.makedirs("data/other")

    for idx in range(nbr_recordings):
        seconds = round(np.random.rand()*1.5 + 0.2, 2)

        rec = sd.rec(frames=int(seconds * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1, device=1)
        sd.wait()  # Wait until recording is finished
        rec = rec.flatten()
        rec = librosa.to_mono(rec)

        file_path = f"data/raw/other/{filename_prefix}_{idx}.wav"
        write(file_path, SAMPLING_RATE, rec)  # Save as WAV file
        print(f"New recording ({seconds}s):", file_path)
        time.sleep(10)
