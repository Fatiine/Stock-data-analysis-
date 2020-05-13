import pandas as pd
from datetime import timedelta


class Predictor:
    """
    Class that take model and horizon as arguments, generate predictions over that time horizon
    """

    def __init__(self, model, horizon):
        """
        Initialize the predictor with the model and the horizon step
        :param model: Classifier model
        :param horizon: int
            Timestep
        """
        self.model = model
        self.horizon = horizon

    def set_parameters(self, threshold, cutoff=0.5):
        """
        Set predictor parameters
        :param threshold: float
        :param cutoff: float
        """
        self.threshold = threshold
        self.cutoff = cutoff

    def set_time_column(self, time_col):
        """
        Set time column name
        :param time_col: str
            Label of time column
        """
        self.time_col = time_col

    def train_test(self, X_train, y_train, X_test, y_test):
        """
        Train the model on X_train, and generate predictions for the X_test
        :param X_train: pandas DataFrame
            Train set features
        :param y_train: pandas DataFrame
            Train set labels
        :param X_test: pandas DataFrame
            Test set features
        :param y_test: pandas DataFrame
            Test set labels
        :return:
            self.model : fitted Model
            pred_df : pandas DataFrame
                Prediction output
        """
        X_train_ = X_train.drop(columns=[self.time_col])
        X_test_ = X_test.drop(columns=[self.time_col])
        self.model.fit(X_train_, y_train)
        pred = self.model.predict(X_test_)
        pred_df = self._create_output_dataframe(X_test, pred)
        print(
            "Accuracy : ",
            (pred_df["signal"].values == y_test.values).mean() * 100,
            " %",
        )
        return self.model, pred_df

    def predict(self, X_test):
        """
        Generate predictions for X_test
        :param X_test: pandas DataFrame
            Features
        :return:
            pred_df : pandas DataFrame
                Prediction output
        """
        X_test_ = X_test.drop(columns=[self.time_col])
        pred = self.model.predict(X_test_)
        pred_df = self._create_output_dataframe(X_test, pred)
        return pred_df

    def _create_output_dataframe(self, features, pred):
        """
        Create output dataframe
        :param features: pandas DataFrame
            Features
        :param pred: pandas DataFrame
            Label predictions
        :return:
            pred_df : pandas DataFrame 
        """
        pred_df = pd.DataFrame(pred, columns=["xgb_output"])
        pred_df["signal"] = pred_df["xgb_output"].apply(
            lambda x: 1 if x > self.cutoff else 0
        )
        pred_df["steps_ahead"] = self.horizon
        pred_df["initial_price"] = features["close_L0"]
        pred_df["prediction"] = round(
            pred_df["initial_price"]
            + pred_df["initial_price"] * pred_df["signal"] * self.threshold / 100,
            5,
        )
        pred_df["prediction_trend"] = pred_df["signal"] * self.threshold
        pred_df["forecast_date"] = features["date"] + timedelta(minutes=self.horizon)
        return pred_df
