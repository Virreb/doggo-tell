import pandas as pd
from src.constants import FOLDERS
from src.models import train_xgboost
from src.dataset import create_features, train_test_split
# train_test_split(test_ratio=0.2)

# train_df = create_features(data_slice="train", augment_loudness=True)
# test_df = create_features(data_slice="test", augment_loudness=False)

train_df = pd.read_pickle("data/train/df.pkl")
test_df = pd.read_pickle("data/test/df.pkl")

# print(train_df.sort_values(by="nbr_frames", ascending=False).head())

y_train = train_df["target"]
y_test = test_df["target"]  # .values()

cols_to_drop_for_training = ["nbr_frames", "used_rate", "file_rate", "file_path", "target"]
x_train = train_df.drop(columns=cols_to_drop_for_training)  # .values()
x_test = test_df.drop(columns=cols_to_drop_for_training)  # .values()

train_xgboost(x_train, x_test, y_train, y_test)

# print(train_df.head())
