import xgboost
from xgboost import DMatrix
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from src.constants import FOLDER_ENCODING


def train_xgboost(x_train, x_test, y_train, y_test):
    print("Training XGBoost")
    params = {
        # "max_depth": 3,
        "objective": "multi:softmax",
        "num_class": 4,
        "n_gpus": 0
    }

    dtrain = DMatrix(data=x_train, label=y_train)
    dtest = DMatrix(data=x_test)

    # xgb_model = XGBClassifier(num_class=4, objective="multi:softmax", use_label_encoder=False)
    # xgb_model.fit(x_train, y_train)

    xgb_model = xgboost.train(params, dtrain)
    y_pred = xgb_model.predict(dtest)
    print(set(y_pred))

    print(" accuracy = ", accuracy_score(y_test, y_pred))
    # print(" f1_score = ", f1_score(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=FOLDER_ENCODING))  # TODO: why doesnt this work?
