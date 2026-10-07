"""5m local S/R rejection scalper aligned with a completed 1h trend."""

from datetime import datetime

from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy import IStrategy, merge_informative_pair


class AggressiveHTFTrendSRScalper(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "5m"
    informative_timeframe = "1h"
    process_only_new_candles = True
    startup_candle_count = 650

    stoploss = -0.06
    minimal_roi = {"0": 0.11}
    trailing_stop = False
    use_exit_signal = False
    position_adjustment_enable = False

    level_window = 12
    touch_atr = 0.18
    minimum_wick_body = 0.75
    minimum_relative_volume = 0.65
    minimum_adx = 15.0
    gap_min = 0.50
    gap_max = 3.00

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
        return [
            (pair, self.informative_timeframe)
            for pair in self.dp.current_whitelist()
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["atr14"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_sma20"] = dataframe["volume"].rolling(20).mean()
        dataframe["relative_volume"] = dataframe["volume"] / dataframe["volume_sma20"]
        dataframe["support"] = dataframe["low"].shift(1).rolling(self.level_window).min()
        dataframe["resistance"] = dataframe["high"].shift(1).rolling(self.level_window).max()
        body = (dataframe["close"] - dataframe["open"]).abs().clip(lower=1e-12)
        dataframe["lower_wick_body"] = (
            dataframe[["open", "close"]].min(axis=1) - dataframe["low"]
        ) / body
        dataframe["upper_wick_body"] = (
            dataframe["high"] - dataframe[["open", "close"]].max(axis=1)
        ) / body

        informative = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.informative_timeframe
        )
        informative["ema20"] = ta.EMA(informative, timeperiod=20)
        informative["ema50"] = ta.EMA(informative, timeperiod=50)
        informative["ema200"] = ta.EMA(informative, timeperiod=200)
        informative["atr14"] = ta.ATR(informative, timeperiod=14)
        informative["adx"] = ta.ADX(informative, timeperiod=14)
        informative["ema50_slope"] = informative["ema50"] - informative["ema50"].shift(4)
        dataframe = merge_informative_pair(
            dataframe,
            informative,
            self.timeframe,
            self.informative_timeframe,
            ffill=True,
        )
        dataframe["long_gap"] = (
            dataframe["ema50_1h"] - dataframe["ema200_1h"]
        ) / dataframe["atr14_1h"]
        dataframe["short_gap"] = -dataframe["long_gap"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        liquid = (
            (dataframe["volume"] > 0)
            & (dataframe["relative_volume"] >= self.minimum_relative_volume)
        )
        support_rejection = (
            (dataframe["low"] <= dataframe["support"] + self.touch_atr * dataframe["atr14"])
            & (dataframe["close"] > dataframe["support"])
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["lower_wick_body"] >= self.minimum_wick_body)
        )
        resistance_rejection = (
            (dataframe["high"] >= dataframe["resistance"] - self.touch_atr * dataframe["atr14"])
            & (dataframe["close"] < dataframe["resistance"])
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["upper_wick_body"] >= self.minimum_wick_body)
        )
        long_trend = (
            (dataframe["ema20"] > dataframe["ema50"])
            & (dataframe["close"] > dataframe["ema200"])
            & (dataframe["close_1h"] > dataframe["ema200_1h"])
            & (dataframe["ema20_1h"] > dataframe["ema50_1h"])
            & (dataframe["ema50_1h"] > dataframe["ema200_1h"])
            & (dataframe["ema50_slope_1h"] > 0)
            & (dataframe["adx_1h"] >= self.minimum_adx)
            & dataframe["long_gap"].between(self.gap_min, self.gap_max)
        )
        short_trend = (
            (dataframe["ema20"] < dataframe["ema50"])
            & (dataframe["close"] < dataframe["ema200"])
            & (dataframe["close_1h"] < dataframe["ema200_1h"])
            & (dataframe["ema20_1h"] < dataframe["ema50_1h"])
            & (dataframe["ema50_1h"] < dataframe["ema200_1h"])
            & (dataframe["ema50_slope_1h"] < 0)
            & (dataframe["adx_1h"] >= self.minimum_adx)
            & dataframe["short_gap"].between(self.gap_min, self.gap_max)
        )
        dataframe.loc[
            support_rejection & long_trend & dataframe["rsi"].between(30, 55) & liquid,
            ["enter_long", "enter_tag"],
        ] = (1, "one_hour_trend_support_long")
        dataframe.loc[
            resistance_rejection & short_trend & dataframe["rsi"].between(45, 70) & liquid,
            ["enter_short", "enter_tag"],
        ] = (1, "one_hour_trend_resistance_short")
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


class AggressiveHTFTrendSRLoose(AggressiveHTFTrendSRScalper):
    level_window = 8
    minimum_relative_volume = 0.50
    minimum_adx = 12.0
    gap_min = 0.0
    gap_max = 4.0


class AggressiveHTFTrendSRStrict(AggressiveHTFTrendSRScalper):
    level_window = 20
    minimum_relative_volume = 0.80
    minimum_adx = 20.0
    gap_min = 0.75
    gap_max = 2.50
