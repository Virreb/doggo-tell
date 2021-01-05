from src.constants import SAMPLING_RATE, NOT_FEATURE_COLUMNS, FEATURE_VECTOR_LENGTH, MAX_FRAMES, NBR_WORKERS, FOLDERS
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

seconds = 2  # Duration of recording

while True:
    print("\nRecording")
    rec = sd.rec(frames=int(seconds * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1, device=1)
    # rec = sd.rec(frames=MAX_FRAMES, samplerate=SAMPLING_RATE, channels=2, device=1)
    sd.wait()  # Wait until recording is finished
    rec = rec.flatten()
    rec = librosa.to_mono(rec)
    # sd.play(rec, samplerate=SAMPLING_RATE)
    # sd.wait()
    features = create_audio_features(rec)

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
    [print(f"{FOLDERS[idx]}: {round(prob, 3)}") for idx, prob in zip(bst.classes_, probas)]

# sd.play(rec, samplerate=SAMPLING_RATE, device=4)  # 4 arctis, 5 speakers
# sd.wait()
# sd.stop()
