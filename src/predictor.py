"""
Predictor — loads saved XGBoost models and generates predictions.
"""
import xgboost as xgb
import numpy as np
import pandas as pd
from config import CLASSIFIER_PATH, REGRESSOR_PATH


class Predictor:
    def __init__(self):
        print(f"Loading classifier from {CLASSIFIER_PATH}")
        self.classifier = xgb.XGBClassifier()
        self.classifier.load_model(str(CLASSIFIER_PATH))

        print(f"Loading regressor from {REGRESSOR_PATH}")
        self.regressor = xgb.XGBRegressor()
        self.regressor.load_model(str(REGRESSOR_PATH))

        print("Models loaded successfully.")

    def predict(self, features_row: pd.Series, feature_cols: list):
        """
        Takes a single row of features, returns (p_up, expected_return).
        
        p_up: probability that price goes up in next 30 min
        expected_return: predicted % return over next 30 min
        """
        # Reshape to 2D array (1 sample × N features)
        X = features_row[feature_cols].values.reshape(1, -1)

        # Handle any NaN — fill with 0 (trees handle this, but just in case)
        X = np.nan_to_num(X, nan=0.0)

        p_up = float(self.classifier.predict_proba(X)[0, 1])
        exp_return = float(self.regressor.predict(X)[0])

        return p_up, exp_return
