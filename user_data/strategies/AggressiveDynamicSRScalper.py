"""Aggressive 5m trend-pullback scalper at dynamic EMA support/resistance."""

from datetime import datetime

from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy import IStrategy, merge_informative_pair


class AggressiveDynamicSRScalper(IStrategy):
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

    dynamic_period = 34
    touch_atr_tolerance = 0.30
    maximum_penetration_atr = 0.60
    minimum_relative_volume = 0.55
    minimum_adx = 15.0

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema34"] = ta.EMA(dataframe, timeperiod=34)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["dynamic_level"] = ta.EMA(dataframe, timeperiod=self.dynamic_period)
        dataframe["atr14"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_sma20"] = dataframe["volume"].rolling(20).mean()
        dataframe["relative_volume"] = dataframe["volume"] / dataframe["volume_sma20"]

        informative = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.informative_timeframe
        )
        informative["ema50"] = ta.EMA(informative, timeperiod=50)
        informative["ema200"] = ta.EMA(informative, timeperiod=200)
        informative["adx"] = ta.ADX(informative, timeperiod=14)
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
        long_touch = (
            (dataframe["low"] <= dataframe["dynamic_level"] + self.touch_atr_tolerance * dataframe["atr14"])
            & (dataframe["low"] >= dataframe["dynamic_level"] - self.maximum_penetration_atr * dataframe["atr14"])
            & (dataframe["close"] > dataframe["dynamic_level"])
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["rsi"].between(38, 58))
            & (dataframe["rsi"] > dataframe["rsi"].shift(1))
        )
        short_touch = (
            (dataframe["high"] >= dataframe["dynamic_level"] - self.touch_atr_tolerance * dataframe["atr14"])
            & (dataframe["high"] <= dataframe["dynamic_level"] + self.maximum_penetration_atr * dataframe["atr14"])
            & (dataframe["close"] < dataframe["dynamic_level"])
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["rsi"].between(42, 62))
            & (dataframe["rsi"] < dataframe["rsi"].shift(1))
        )
        long_trend = (
            (dataframe["ema20"] > dataframe["ema50"])
            & (dataframe["close_15m"] > dataframe["ema200_15m"])
            & (dataframe["ema50_15m"] > dataframe["ema200_15m"])
            & (dataframe["adx_15m"] >= self.minimum_adx)
        )
        short_trend = (
            (dataframe["ema20"] < dataframe["ema50"])
            & (dataframe["close_15m"] < dataframe["ema200_15m"])
            & (dataframe["ema50_15m"] < dataframe["ema200_15m"])
            & (dataframe["adx_15m"] >= self.minimum_adx)
        )

        dataframe.loc[long_touch & long_trend & liquid, ["enter_long", "enter_tag"]] = (
            1,
            f"ema{self.dynamic_period}_support_long",
        )
        dataframe.loc[short_touch & short_trend & liquid, ["enter_short", "enter_tag"]] = (
            1,
            f"ema{self.dynamic_period}_resistance_short",
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


class AggressiveDynamicSR20(AggressiveDynamicSRScalper):
    dynamic_period = 20


class AggressiveDynamicSR50(AggressiveDynamicSRScalper):
    dynamic_period = 50
