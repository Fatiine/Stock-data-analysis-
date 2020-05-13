import pandas as pd
import numpy as np


class FeaturesGenerator:
    """
    A class that generates features for a stock data
    """

    REGRESSORS = ["high", "low", "close", "volume"]
    TIME_UNIT_DICO = {"D": 60 * 24, "H": 60, "M": 1}
    INDICATORS_DICO = {"simple_mavg": [5, 10, 25]}

    # TODO : Add other indicators : MACD , RSI
    # 'macd_12_26' : [12,26],
    # "macd_5_35": [5,35]}
    def __init__(self, df):
        """
        Initialize the FeaturesGenerator with the stock dataframe of a pair
        :param df: pandas DataFrame
            DataFrame representing the stock data of a pair
        """
        self.df = df

    def set_columns(self, time_col, price_col):
        """
        Set time and price columns to the DataFrame
        :param time_col: str
            Name of the column containing the date/time
        :param price_col: str
            Name of the column that represents the price at closure time
        """
        self.time_col = time_col
        self.price_col = price_col
        self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])

    def preprocess(self, threshold):
        """
        Enhance the dataframe with indicators and price features
        :param threshold: float
        """
        self.df = self.df.sort_values(by=self.time_col).reset_index(drop=True)
        # if price is NA, fill with the previous valid number
        self.df[self.price_col] = self.df[self.price_col].fillna(method="ffill")
        self._create_timesteps_indexes()
        self._create_indexes_minus()
        self._get_price_differences()
        self._get_indicators(threshold=threshold)
        self._get_indicators_over_timesteps()

    def generate_buy_signal(self, lowest_dip):
        """
        Create buy signal from data features
        :return:
        features : pandas DataFrame
            The generated features from the preprocessing step
        labels : pandas Series
            0 means we don't buy
            1 means we buy
        """
        self.df["buy_signal"] = self.df.apply(
            lambda x: 1
            if (
                not pd.isna(x["mins_to_fill"])
                and (x["mins_to_fill"] <= 30)
                and (x["lowest_dip_b4_sell_perc"] >= lowest_dip)
            )
            else 0,
            axis=1,
        )
        return self.df.drop(columns="buy_signal"), self.df["buy_signal"]

    def generate_sell_signal(self):
        """
        Create sell signal from data elements
        :return:
        features : pandas DataFrame
            The generated features from the preprocessing step
        labels : pandas Series
            0 means we don't sell
            1 means we sell
        """
        pass

    def create_lags(self, regressors, max_lag):
        """
        Adds "max_lag" lags of the regressors' columns
        :param regressors: list of str
            List of names of columns that represents regressors
        :param max_lag: int
            Number of lags to create
        """
        for i in range(max_lag + 1):
            for el in regressors:
                self.df[el + "_L" + str(i)] = self.df[el].shift(i)

    def create_regressors(self):
        regressors = [
            "date",
            "pair",
            "exchange",
            "open",
            "high",
            "low",
            "close",
            # "buy_signal"
        ]
        lag_features = [col for col in self.df.columns if "_L" in col]
        regressors += lag_features
        return self.df[regressors]

    def _create_timesteps_indexes(self, timesteps=[30, 60, 180], time_unit="M"):
        self.df["idx"] = range(1, len(self.df) + 1)
        for t in timesteps:
            self.df["index_" + str(t)] = (len(self.df) - self.df["idx"]) // (
                t * self.TIME_UNIT_DICO[time_unit]
            ) + 1

    def _create_indexes_minus(self, minus_period=[1, 2], time_unit="D"):
        for p in minus_period:
            self.df["idx_min_" + str(p) + time_unit.lower()] = (
                self.df["idx"] - p * self.TIME_UNIT_DICO[time_unit]
            )

    def _get_price_differences(self):
        self.df["last_price_locf_diff"] = self.df[self.price_col].diff()
        self.df["low_high_spread"] = self.df["low"] - self.df["high"]
        self.df["open_close_spread"] = self.df["close"] - self.df["open"]

    def _get_indicators(self, threshold):
        """
        Compute technical indicators for different time intervals.
        :param threshold : float
        """
        self.df["price_required"] = self.df[self.price_col] * (1 + threshold / 100)

        # SIMPLE MOVING AVERAGE OVER THE TIMESTEPS : self.INDICATORS_DICO["simple_mavg"]
        indicator = "simple_mavg"
        for i in self.INDICATORS_DICO[indicator]:
            self.df[indicator + "_" + str(i)] = (
                self.df[self.price_col].rolling(window=i, min_periods=1).mean()
            )
            self.df[indicator + "_" + str(i) + "_diff"] = self.df[
                indicator + "_" + str(i)
            ].diff()
            self.df[indicator + "_" + str(i) + "_doub_diff"] = self.df[
                indicator + "_" + str(i) + "_diff"
            ].diff()

        # TODO: The following line should be optimized , complexity O(n^2)
        self.df["mins_to_fill"] = self.df.apply(
            lambda x: self.df[self.df.idx > x.idx][self.df.high >= x.price_required]
            .iloc[0]
            .idx
            - x.idx
            if len(self.df[self.df.idx > x.idx][self.df.high >= x.price_required]) > 0
            else np.nan,
            axis=1,
        )
        # self.data["max_high_1d"] = self.data.apply(lambda x : max(self.data[x.idx>self.data.idx_min_1d][self.data.idx_min_1d<x.idx_min_1d]['high']) if len(self.data[x.idx>self.data.idx_min_1d][self.data.idx_min_1d<x.idx_min_1d]) > 0 else np.nan,axis=1)

        self.df["idx_p_mins_to_fill"] = self.df.idx + self.df.mins_to_fill
        self.df["lowest_dip_b4_sell"] = self.df.apply(
            lambda x: min(
                self.df[self.df.idx > x.idx][self.df.idx <= x.idx_p_mins_to_fill]["low"]
            )
            if len(self.df[self.df.idx > x.idx][self.df.idx <= x.idx_p_mins_to_fill])
            > 0
            else np.nan,
            axis=1,
        )
        self.df["lowest_dip_b4_sell_perc"] = (
            (self.df.lowest_dip_b4_sell - self.df[self.price_col])
            / self.df[self.price_col]
        ) * 100

    def _get_indicators_over_timesteps(self, timesteps=[30, 60, 180]):
        for t in timesteps:
            max_high = self.df.groupby(["index_" + str(t)])["high"].max().shift(-1)
            min_low = self.df.groupby(["index_" + str(t)])["low"].min().shift(-1)
            last_close = self.df.groupby(["index_" + str(t)])["close"].last().shift(-1)
            first_open = self.df.groupby(["index_" + str(t)])["open"].first().shift(-1)
            sum_volume = self.df.groupby(["index_" + str(t)])["volume"].sum().shift(-1)

            self.df = self.df.merge(
                max_high, how="left", on="index_" + str(t), suffixes=("", "_" + str(t))
            )
            self.df = self.df.merge(
                min_low, how="left", on="index_" + str(t), suffixes=("", "_" + str(t))
            )
            self.df = self.df.merge(
                last_close,
                how="left",
                on="index_" + str(t),
                suffixes=("", "_" + str(t)),
            )
            self.df = self.df.merge(
                first_open,
                how="left",
                on="index_" + str(t),
                suffixes=("", "_" + str(t)),
            )
            self.df = self.df.merge(
                sum_volume,
                how="left",
                on="index_" + str(t),
                suffixes=("", "_" + str(t)),
            )

            self.df["low_high_spread_" + str(t)] = (
                self.df["low_" + str(t)] - self.df["high_" + str(t)]
            )
            self.df["open_close_spread_" + str(t)] = (
                self.df["close_" + str(t)] - self.df["open_" + str(t)]
            )

            indicator = "simple_mavg"
            for j in self.INDICATORS_DICO[indicator]:
                self.df[indicator + "_" + str(j) + "_" + str(t)] = (
                    self.df["close_" + str(t)].rolling(window=j, min_periods=1).mean()
                )
                self.df[indicator + "_" + str(j) + "_diff" + "_" + str(t)] = self.df[
                    indicator + "_" + str(j) + "_" + str(t)
                ].diff()
                self.df[
                    indicator + "_" + str(j) + "_doub_diff" + "_" + str(t)
                ] = self.df[indicator + "_" + str(j) + "_diff" + "_" + str(t)].diff()
