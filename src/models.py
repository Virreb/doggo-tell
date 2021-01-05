from src.dataset import train_test_split, create_features
from xgboost import DMatrix, XGBClassifier
from sklearn.metrics import classification_report
from src.constants import FOLDER_ENCODING, NOT_FEATURE_COLUMNS, NBR_WORKERS, FOLDERS
import pickle


def train_xgboost(x_train, x_test, y_train, y_test):
    print("Training XGBoost")
    params = {
        # "max_depth": 3,
        "nthread": NBR_WORKERS,
        "objective": "multi:softmax",
        "num_class": len(FOLDERS),
        "n_gpus": 0
    }

    dtrain = DMatrix(data=x_train, label=y_train)
    dtest = DMatrix(data=x_test)

    bst = XGBClassifier(num_class=4, objective="multi:softmax", use_label_encoder=False)
    bst.fit(x_train, y_train)
    y_pred = bst.predict(x_test)

    # bst = xgb.train(params, dtrain, num_boost_round=10)
    # y_pred = bst.predict(dtest)

    # print(" accuracy = ", accuracy_score(y_test, y_pred))
    # print(" f1_score = ", f1_score(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=FOLDER_ENCODING))
    # bst.save_model("artifacts/xgboost.model")
    pickle.dump(bst, open("artifacts/xgboost.model", "wb"))


if __name__ == "__main__":

    import pandas as pd

    train_test_split(test_ratio=0.2)

    train_df = create_features(data_slice="train", augment_loudness=True)
    test_df = create_features(data_slice="test", augment_loudness=False)

    train_df = pd.read_pickle("data/train/df.pkl")
    test_df = pd.read_pickle("data/test/df.pkl")

    y_train = train_df["target"]
    y_test = test_df["target"]

    x_train = train_df.drop(columns=NOT_FEATURE_COLUMNS)
    x_test = test_df.drop(columns=NOT_FEATURE_COLUMNS)

    train_xgboost(x_train, x_test, y_train, y_test)
