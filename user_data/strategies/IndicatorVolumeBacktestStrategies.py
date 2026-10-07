"""Isolated indicator + volume strategies for apples-to-apples backtests.

Each concrete strategy intentionally has only one entry-signal family.  They
share the same execution, risk, trailing-stop and leverage rules so the
backtest comparison is not contaminated by unrelated entry filters.
"""

from datetime import datetime

import numpy as np
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy import IStrategy
from technical import qtpylib


class IndicatorVolumeBacktestBase(IStrategy):
    INTERFACE_VERSION = 3

    can_short = True
    timeframe = "15m"
    process_only_new_candles = True
    startup_candle_count = 220

    minimal_roi = {}
    stoploss = -0.20
    trailing_stop = True
    trailing_stop_positive = 0.10
    trailing_stop_positive_offset = 0.30
    trailing_only_offset_is_reached = True

    use_exit_signal = False
    position_adjustment_enable = False

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["volume_sma20"] = dataframe["volume"].rolling(20).mean()
        dataframe["relative_volume"] = dataframe["volume"] / dataframe["volume_sma20"]
        return dataframe

    @staticmethod
    def volume_confirmed(dataframe: DataFrame):
        return (
            (dataframe["volume"] > 0)
            & (dataframe["volume_sma20"] > 0)
            & (dataframe["relative_volume"] >= 1.0)
        )

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # All variants use only the identical initial stop and trailing stop.
        return dataframe

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag,
        side: str,
        **kwargs,
    ) -> float:
        if pair == "BTC/USDT:USDT":
            return min(100.0, max_leverage)
        if pair == "ETH/USDT:USDT":
            return min(75.0, max_leverage)
        return 1.0


class RSIExtremeVolumeStrategy(IndicatorVolumeBacktestBase):
    """Exact RSI(14) endpoints requested by the user, confirmed by volume."""

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        volume_ok = self.volume_confirmed(dataframe)
        dataframe.loc[
            (dataframe["rsi"] <= 0.0) & volume_ok,
            ["enter_long", "enter_tag"],
        ] = (1, "rsi_0_volume")
        dataframe.loc[
            (dataframe["rsi"] >= 100.0) & volume_ok,
            ["enter_short", "enter_tag"],
        ] = (1, "rsi_100_volume")
        return dataframe


class TripleEMAVolumeStrategy(IndicatorVolumeBacktestBase):
    """EMA 13/21 crossover with EMA 200 regime and volume confirmation."""

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        dataframe["ema13"] = ta.EMA(dataframe, timeperiod=13)
        dataframe["ema21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        volume_ok = self.volume_confirmed(dataframe)
        dataframe.loc[
            qtpylib.crossed_above(dataframe["ema13"], dataframe["ema21"])
            & (dataframe["ema21"] > dataframe["ema200"])
            & (dataframe["close"] > dataframe["ema200"])
            & volume_ok,
            ["enter_long", "enter_tag"],
        ] = (1, "ema13_cross_21_above_200_volume")
        dataframe.loc[
            qtpylib.crossed_below(dataframe["ema13"], dataframe["ema21"])
            & (dataframe["ema21"] < dataframe["ema200"])
            & (dataframe["close"] < dataframe["ema200"])
            & volume_ok,
            ["enter_short", "enter_tag"],
        ] = (1, "ema13_cross_21_below_200_volume")
        return dataframe


class StochasticVolumeStrategy(IndicatorVolumeBacktestBase):
    """Stochastic 5,3,3 reversal at conventional extremes plus volume."""

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        stochastic = ta.STOCH(
            dataframe,
            fastk_period=5,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0,
        )
        dataframe["stoch_k"] = stochastic["slowk"]
        dataframe["stoch_d"] = stochastic["slowd"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        volume_ok = self.volume_confirmed(dataframe)
        dataframe.loc[
            qtpylib.crossed_above(dataframe["stoch_k"], dataframe["stoch_d"])
            & (dataframe["stoch_k"] <= 20)
            & (dataframe["stoch_d"] <= 20)
            & volume_ok,
            ["enter_long", "enter_tag"],
        ] = (1, "stoch_oversold_cross_volume")
        dataframe.loc[
            qtpylib.crossed_below(dataframe["stoch_k"], dataframe["stoch_d"])
            & (dataframe["stoch_k"] >= 80)
            & (dataframe["stoch_d"] >= 80)
            & volume_ok,
            ["enter_short", "enter_tag"],
        ] = (1, "stoch_overbought_cross_volume")
        return dataframe


class RepeatedOrderblockVolumeStrategy(IndicatorVolumeBacktestBase):
    """Trade the third distinct rejection of a causal displacement orderblock."""

    orderblock_valid_candles = 96
    minimum_touches = 3

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        dataframe["atr14"] = ta.ATR(dataframe, timeperiod=14)

        body = (dataframe["close"] - dataframe["open"]).abs()
        bullish_displacement = (
            (dataframe["close"] > dataframe["open"])
            & (body > dataframe["atr14"])
            & (dataframe["close"] > dataframe["high"].shift(1))
            & (dataframe["relative_volume"] >= 1.5)
            & (dataframe["close"].shift(1) < dataframe["open"].shift(1))
        )
        bearish_displacement = (
            (dataframe["close"] < dataframe["open"])
            & (body > dataframe["atr14"])
            & (dataframe["close"] < dataframe["low"].shift(1))
            & (dataframe["relative_volume"] >= 1.5)
            & (dataframe["close"].shift(1) > dataframe["open"].shift(1))
        )

        bull_low = np.where(bullish_displacement, dataframe["low"].shift(1), np.nan)
        bull_high = np.where(bullish_displacement, dataframe["open"].shift(1), np.nan)
        bear_low = np.where(bearish_displacement, dataframe["open"].shift(1), np.nan)
        bear_high = np.where(bearish_displacement, dataframe["high"].shift(1), np.nan)

        n = len(dataframe)
        bull_touch_count = np.zeros(n, dtype=np.int16)
        bear_touch_count = np.zeros(n, dtype=np.int16)
        bull_rejection = np.zeros(n, dtype=np.int8)
        bear_rejection = np.zeros(n, dtype=np.int8)

        active_bull = None
        active_bear = None
        previous_bull_intersection = False
        previous_bear_intersection = False

        opens = dataframe["open"].to_numpy()
        highs = dataframe["high"].to_numpy()
        lows = dataframe["low"].to_numpy()
        closes = dataframe["close"].to_numpy()

        for i in range(n):
            # A block detected on displacement candle i becomes available only
            # from candle i+1, after all its defining data is closed.
            if i > 0 and not np.isnan(bull_low[i - 1]):
                active_bull = [float(bull_low[i - 1]), float(bull_high[i - 1]), i, 0]
                previous_bull_intersection = False
            if i > 0 and not np.isnan(bear_low[i - 1]):
                active_bear = [float(bear_low[i - 1]), float(bear_high[i - 1]), i, 0]
                previous_bear_intersection = False

            if active_bull is not None:
                zone_low, zone_high, born, touches = active_bull
                if i - born >= self.orderblock_valid_candles or closes[i] < zone_low:
                    active_bull = None
                    previous_bull_intersection = False
                else:
                    intersects = lows[i] <= zone_high and highs[i] >= zone_low
                    if intersects and not previous_bull_intersection:
                        touches += 1
                        active_bull[3] = touches
                    bull_touch_count[i] = touches
                    bull_rejection[i] = int(
                        intersects
                        and touches >= self.minimum_touches
                        and closes[i] > zone_high
                        and closes[i] > opens[i]
                    )
                    previous_bull_intersection = intersects

            if active_bear is not None:
                zone_low, zone_high, born, touches = active_bear
                if i - born >= self.orderblock_valid_candles or closes[i] > zone_high:
                    active_bear = None
                    previous_bear_intersection = False
                else:
                    intersects = highs[i] >= zone_low and lows[i] <= zone_high
                    if intersects and not previous_bear_intersection:
                        touches += 1
                        active_bear[3] = touches
                    bear_touch_count[i] = touches
                    bear_rejection[i] = int(
                        intersects
                        and touches >= self.minimum_touches
                        and closes[i] < zone_low
                        and closes[i] < opens[i]
                    )
                    previous_bear_intersection = intersects

        dataframe["bull_ob_touch_count"] = bull_touch_count
        dataframe["bear_ob_touch_count"] = bear_touch_count
        dataframe["bull_ob_rejection"] = bull_rejection
        dataframe["bear_ob_rejection"] = bear_rejection
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        volume_ok = self.volume_confirmed(dataframe)
        dataframe.loc[
            (dataframe["bull_ob_rejection"] == 1) & volume_ok,
            ["enter_long", "enter_tag"],
        ] = (1, "bull_ob_third_touch_reversal_volume")
        dataframe.loc[
            (dataframe["bear_ob_rejection"] == 1) & volume_ok,
            ["enter_short", "enter_tag"],
        ] = (1, "bear_ob_third_touch_reversal_volume")
        return dataframe
