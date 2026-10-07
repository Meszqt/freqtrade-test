"""Causal 5m EMA200 support/resistance reaction scalper.

Each concrete subclass tests one higher timeframe independently.  Freqtrade's
``merge_informative_pair`` shifts the informative candle before forward-filling
it, so the 5m signal only sees an EMA200 calculated from completed candles.

The signal first observes a wick through/near the higher-timeframe EMA200 and
a close back on the trend side.  The resulting limit order is placed at the
EMA200 itself, making the simulated entry match the level being tested.
"""

from datetime import datetime

import pandas as pd
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy import IStrategy, merge_informative_pair


class MTFEMA200ReactionScalper(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True

    timeframe = "5m"
    informative_timeframe = "1h"
    process_only_new_candles = True
    startup_candle_count = 300

    # At 10x leverage and the configured 0.05% fee per side, completed ROI and
    # stop trades are approximately +11% and -7%.  Net reward:risk ~= 1.57.
    stoploss = -0.06
    minimal_roi = {"0": 0.12}
    trailing_stop = False
    use_exit_signal = False
    position_adjustment_enable = False

    leverage_value = 10.0
    slope_lookback = 3
    minimum_informative_adx = 15.0
    minimum_relative_volume = 0.60
    touch_allowance_atr = 0.12
    maximum_overshoot_atr = 0.55
    minimum_wick_fraction = 0.28
    minimum_close_location = 0.60
    maximum_level_distance_atr = 0.55
    approach_distance_atr = 0.85
    require_confirmed_rejection = False

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
        dataframe["ema9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["atr14"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["rsi14"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_mean20"] = dataframe["volume"].rolling(20).mean()
        dataframe["relative_volume"] = (
            dataframe["volume"] / dataframe["volume_mean20"]
        )

        candle_range = (dataframe["high"] - dataframe["low"]).clip(lower=1e-12)
        dataframe["close_location"] = (
            (dataframe["close"] - dataframe["low"]) / candle_range
        )
        dataframe["lower_wick_fraction"] = (
            dataframe[["open", "close"]].min(axis=1) - dataframe["low"]
        ) / candle_range
        dataframe["upper_wick_fraction"] = (
            dataframe["high"] - dataframe[["open", "close"]].max(axis=1)
        ) / candle_range

        informative = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.informative_timeframe
        )
        informative["ema200"] = ta.EMA(informative, timeperiod=200)
        informative["ema50"] = ta.EMA(informative, timeperiod=50)
        informative["atr14"] = ta.ATR(informative, timeperiod=14)
        informative["adx14"] = ta.ADX(informative, timeperiod=14)
        informative["rsi14"] = ta.RSI(informative, timeperiod=14)
        informative["ema200_slope"] = (
            informative["ema200"]
            - informative["ema200"].shift(self.slope_lookback)
        ) / informative["ema200"].shift(self.slope_lookback)

        return merge_informative_pair(
            dataframe,
            informative,
            self.timeframe,
            self.informative_timeframe,
            ffill=True,
        )

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        suffix = self.informative_timeframe
        level = dataframe[f"ema200_{suffix}"]
        informative_atr = dataframe[f"atr14_{suffix}"]
        informative_close = dataframe[f"close_{suffix}"]
        informative_ema50 = dataframe[f"ema50_{suffix}"]
        informative_adx = dataframe[f"adx14_{suffix}"]
        informative_rsi = dataframe[f"rsi14_{suffix}"]
        slope = dataframe[f"ema200_slope_{suffix}"]

        # Use the tighter 5m ATR to define a tradeable touch.  A proportional
        # cap also prevents a stale higher-timeframe level from being chased.
        touch_buffer = self.touch_allowance_atr * dataframe["atr14"]
        overshoot_buffer = self.maximum_overshoot_atr * dataframe["atr14"]
        maximum_distance = self.maximum_level_distance_atr * dataframe["atr14"]

        liquid = (
            (dataframe["volume"] > 0)
            & (dataframe["relative_volume"] >= self.minimum_relative_volume)
            & dataframe["atr14"].notna()
            & informative_atr.notna()
        )

        rising_support = (
            (slope > 0)
            & (informative_close > level)
            & (informative_adx >= self.minimum_informative_adx)
            & (informative_rsi >= 45)
        )
        falling_resistance = (
            (slope < 0)
            & (informative_close < level)
            & (informative_adx >= self.minimum_informative_adx)
            & (informative_rsi <= 55)
        )

        # In live trading an exact-level entry and a same-candle confirmed
        # rejection cannot both be known causally.  The default therefore
        # submits a resting limit while price is moving toward the known EMA.
        # The rejection is then what must happen after the order fills.  The
        # confirmed mode remains available for conservative comparisons.
        long_approach = (
            (dataframe["close"] > level)
            & ((dataframe["close"] - level) <= self.approach_distance_atr * dataframe["atr14"])
            & (dataframe["close"] < dataframe["close"].shift(1))
            & (dataframe["low"] <= dataframe["low"].shift(1))
            & (dataframe["ema9"] <= dataframe["ema9"].shift(1))
            & dataframe["rsi14"].between(34, 58)
        )
        short_approach = (
            (dataframe["close"] < level)
            & ((level - dataframe["close"]) <= self.approach_distance_atr * dataframe["atr14"])
            & (dataframe["close"] > dataframe["close"].shift(1))
            & (dataframe["high"] >= dataframe["high"].shift(1))
            & (dataframe["ema9"] >= dataframe["ema9"].shift(1))
            & dataframe["rsi14"].between(42, 66)
        )

        bullish_rejection = (
            (dataframe["low"] <= level + touch_buffer)
            & (dataframe["low"] >= level - overshoot_buffer)
            & (dataframe["close"] > level)
            & ((dataframe["close"] - level) <= maximum_distance)
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["close"] > dataframe["close"].shift(1))
            & (dataframe["close_location"] >= self.minimum_close_location)
            & (dataframe["lower_wick_fraction"] >= self.minimum_wick_fraction)
            & (dataframe["ema9"] >= dataframe["ema9"].shift(1))
            & dataframe["rsi14"].between(40, 68)
        )
        bearish_rejection = (
            (dataframe["high"] >= level - touch_buffer)
            & (dataframe["high"] <= level + overshoot_buffer)
            & (dataframe["close"] < level)
            & ((level - dataframe["close"]) <= maximum_distance)
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["close"] < dataframe["close"].shift(1))
            & (dataframe["close_location"] <= 1.0 - self.minimum_close_location)
            & (dataframe["upper_wick_fraction"] >= self.minimum_wick_fraction)
            & (dataframe["ema9"] <= dataframe["ema9"].shift(1))
            & dataframe["rsi14"].between(32, 60)
        )

        long_setup = bullish_rejection if self.require_confirmed_rejection else long_approach
        short_setup = bearish_rejection if self.require_confirmed_rejection else short_approach

        # One order per setup cluster.  A later, distinct approach can create
        # another trade, which keeps the implementation suitable for scalping.
        first_long_setup = long_setup & ~long_setup.shift(1).fillna(False)
        first_short_setup = short_setup & ~short_setup.shift(1).fillna(False)

        dataframe.loc[
            rising_support & first_long_setup & liquid,
            ["enter_long", "enter_tag"],
        ] = (1, f"{suffix}_ema200_support")
        dataframe.loc[
            falling_resistance & first_short_setup & liquid,
            ["enter_short", "enter_tag"],
        ] = (1, f"{suffix}_ema200_resistance")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_entry_price(
        self,
        pair: str,
        trade,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """Keep every filled order at the selected completed-candle EMA200."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_rate
        level = dataframe.iloc[-1].get(f"ema200_{self.informative_timeframe}")
        if level is None or pd.isna(level) or float(level) <= 0:
            return proposed_rate
        return float(level)

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
        return min(self.leverage_value, max_leverage)


class MTFEMA200Reaction1h(MTFEMA200ReactionScalper):
    informative_timeframe = "1h"


class MTFEMA200Reaction2h(MTFEMA200ReactionScalper):
    informative_timeframe = "2h"


class MTFEMA200Reaction4h(MTFEMA200ReactionScalper):
    informative_timeframe = "4h"


class MTFEMA200Reaction8h(MTFEMA200ReactionScalper):
    informative_timeframe = "8h"


class MTFEMA200Reaction12h(MTFEMA200ReactionScalper):
    informative_timeframe = "12h"


class MTFEMA200Reaction1d(MTFEMA200ReactionScalper):
    informative_timeframe = "1d"


class MTFEMA200Confirmed1h(MTFEMA200ReactionScalper):
    """Conservative control: enter only if EMA is retested after rejection."""

    informative_timeframe = "1h"
    require_confirmed_rejection = True
