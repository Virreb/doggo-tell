from src.constants import SAMPLING_RATE, NOT_FEATURE_COLUMNS, NBR_WORKERS, FOLDERS, MAX_RAW_SOUND_POINTS
import sounddevice as sd
from xgboost import DMatrix, XGBClassifier
import xgboost as xgb
from src.dataset import create_audio_features
import librosa
import numpy as np
import pandas as pd
import pickle

# bst = XGBClassifier(n_jobs=NBR_WORKERS).load_model("artifacts/xgboost.model")

bst = pickle.load(open("artifacts/xgboost.model", "rb"))
# bst = xgb.Booster({"nthread": NBR_WORKERS})
# bst.load_model("artifacts/xgboost.model")
sound_level_threshold = 0.2

print("Started Doggo-tell")
while True:
    # rec = sd.rec(frames=int(seconds * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1, device=1)
    rec = sd.rec(frames=MAX_RAW_SOUND_POINTS, samplerate=SAMPLING_RATE, channels=1, device=1)
    # rec = sd.rec(frames=MAX_FRAMES, samplerate=SAMPLING_RATE, channels=2, device=1)
    sd.wait()  # Wait until recording is finished
    rec = rec.flatten()
    rec = librosa.to_mono(rec)
    # sd.play(rec, samplerate=SAMPLING_RATE)
    # sd.wait()
    sound_level = np.linalg.norm(rec)
    if sound_level >= sound_level_threshold:
        features = create_audio_features(rec)   # max MAX_RAW_SOUND_POINTS of len here

        # for key in NOT_FEATURE_COLUMNS:
        #     if key in features:
        #         features.pop(key)

        # x = np.fromiter(features.values(), dtype=float)
        df = pd.DataFrame([features])
        x = df.drop(columns=[col for col in NOT_FEATURE_COLUMNS if col in df.columns])

        # x = x.reshape(1, x.shape[0])
        # dpred = DMatrix(data=x)
        pred = bst.predict(x)
        probas = bst.predict_proba(x).flatten()
        print("Sound level:", round(sound_level, 3))
        [print(f"{FOLDERS[idx]}: {round(prob, 3)}") for idx, prob in zip(bst.classes_, probas)]
        print("\n")
    print("\n")
