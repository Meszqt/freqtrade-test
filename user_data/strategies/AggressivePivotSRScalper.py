"""Aggressive 5m scalper using confirmed pivot support/resistance and 15m trend."""

from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair
from technical import qtpylib


class AggressivePivotSRScalper(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "5m"
    informative_timeframe = "15m"
    process_only_new_candles = True
    startup_candle_count = 650

    stoploss = -0.06
    minimal_roi = {"0": 0.11}
    trailing_stop = False
    use_exit_signal = False
    position_adjustment_enable = False

    pivot_confirmation = 2
    maximum_pivot_age = 72
    touch_atr_tolerance = 0.20
    wick_body_ratio = 1.0
    minimum_relative_volume = 0.80
    minimum_adx = 16.0
    minimum_room_atr = 1.5

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    def informative_pairs(self):
        if not self.dp:
            return []
        return [(pair, self.informative_timeframe) for pair in self.dp.current_whitelist()]

    @staticmethod
    def _informative_indicators(dataframe: DataFrame) -> DataFrame:
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["atr14"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_sma20"] = dataframe["volume"].rolling(20).mean()
        dataframe["relative_volume"] = dataframe["volume"] / dataframe["volume_sma20"]

        width = self.pivot_confirmation * 2 + 1
        candidate_low = dataframe["low"].shift(self.pivot_confirmation)
        candidate_high = dataframe["high"].shift(self.pivot_confirmation)
        pivot_low = candidate_low.where(candidate_low == dataframe["low"].rolling(width).min())
        pivot_high = candidate_high.where(candidate_high == dataframe["high"].rolling(width).max())
        dataframe["support"] = pivot_low.ffill()
        dataframe["resistance"] = pivot_high.ffill()

        row_number = pd.Series(np.arange(len(dataframe)), index=dataframe.index, dtype=float)
        last_support_row = row_number.where(pivot_low.notna()).ffill()
        last_resistance_row = row_number.where(pivot_high.notna()).ffill()
        dataframe["support_age"] = row_number - last_support_row
        dataframe["resistance_age"] = row_number - last_resistance_row

        body = (dataframe["close"] - dataframe["open"]).abs().clip(lower=1e-12)
        dataframe["lower_wick_ratio"] = (
            dataframe[["open", "close"]].min(axis=1) - dataframe["low"]
        ) / body
        dataframe["upper_wick_ratio"] = (
            dataframe["high"] - dataframe[["open", "close"]].max(axis=1)
        ) / body

        informative = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.informative_timeframe
        )
        informative = self._informative_indicators(informative)
        return merge_informative_pair(
            dataframe,
            informative,
            self.timeframe,
            self.informative_timeframe,
            ffill=True,
        )

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        liquid = (
            (dataframe["volume"] > 0)
            & (dataframe["relative_volume"] >= self.minimum_relative_volume)
        )
        long_rejection = (
            (dataframe["support_age"].between(2, self.maximum_pivot_age))
            & (dataframe["low"] <= dataframe["support"] + self.touch_atr_tolerance * dataframe["atr14"])
            & (dataframe["close"] > dataframe["support"])
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["lower_wick_ratio"] >= self.wick_body_ratio)
            & qtpylib.crossed_above(dataframe["rsi"], 35)
            & ((dataframe["resistance"] - dataframe["close"]) >= self.minimum_room_atr * dataframe["atr14"])
        )
        short_rejection = (
            (dataframe["resistance_age"].between(2, self.maximum_pivot_age))
            & (dataframe["high"] >= dataframe["resistance"] - self.touch_atr_tolerance * dataframe["atr14"])
            & (dataframe["close"] < dataframe["resistance"])
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["upper_wick_ratio"] >= self.wick_body_ratio)
            & qtpylib.crossed_below(dataframe["rsi"], 65)
            & ((dataframe["close"] - dataframe["support"]) >= self.minimum_room_atr * dataframe["atr14"])
        )

        trend_long = (
            (dataframe["close_15m"] > dataframe["ema200_15m"])
            & (dataframe["ema50_15m"] > dataframe["ema200_15m"])
            & (dataframe["adx_15m"] >= self.minimum_adx)
        )
        trend_short = (
            (dataframe["close_15m"] < dataframe["ema200_15m"])
            & (dataframe["ema50_15m"] < dataframe["ema200_15m"])
            & (dataframe["adx_15m"] >= self.minimum_adx)
        )

        dataframe.loc[long_rejection & trend_long & liquid, ["enter_long", "enter_tag"]] = (
            1,
            "pivot_support_long",
        )
        dataframe.loc[short_rejection & trend_short & liquid, ["enter_short", "enter_tag"]] = (
            1,
            "pivot_resistance_short",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return min(10.0, max_leverage)


class AggressivePivotSRScalperLoose(AggressivePivotSRScalper):
    wick_body_ratio = 0.6
    minimum_relative_volume = 0.55
    minimum_adx = 12.0
    minimum_room_atr = 1.0


class AggressivePivotSRScalperStrict(AggressivePivotSRScalper):
    wick_body_ratio = 1.5
    minimum_relative_volume = 1.0
    minimum_adx = 20.0
    minimum_room_atr = 2.0
