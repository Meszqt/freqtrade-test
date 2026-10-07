"""Causal 5 minute support/resistance scalper research variants.

The signal treats the lowest/highest price of the preceding short lookback as
support/resistance, waits for an intrabar sweep of that level, and only enters
after the candle closes back inside the range.  A 15 minute regime filter keeps
the trades aligned with the broader trend.  All level calculations are shifted
by one candle, so no future candle participates in an entry decision.
"""

from datetime import datetime

import numpy as np
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair


class AlternativeSRMomentumScalper(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "5m"
    informative_timeframe = "15m"
    process_only_new_candles = True
    startup_candle_count = 650

    # At 10x, a normal stop loses about 4% after two 0.05% fees, while a
    # normal ROI exit earns 6%.  Thus the realised net reward:risk is 1.5:1.
    stoploss = -0.03
    minimal_roi = {"0": 0.06}
    trailing_stop = False
    use_exit_signal = False
    position_adjustment_enable = False

    level_window = 12
    rsi2_extreme = 10.0
    minimum_adx = 12.0
    minimum_relative_volume = 0.55
    sweep_atr = 0.12
    require_fast_ema_reclaim = True

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
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["atr14"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["ema9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi2"] = ta.RSI(dataframe, timeperiod=2)
        dataframe["volume_sma20"] = dataframe["volume"].rolling(20).mean()
        dataframe["relative_volume"] = dataframe["volume"] / dataframe["volume_sma20"]

        # Current candle is deliberately excluded from both structural levels.
        dataframe["support"] = (
            dataframe["low"].shift(1).rolling(self.level_window).min()
        )
        dataframe["resistance"] = (
            dataframe["high"].shift(1).rolling(self.level_window).max()
        )

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
        fast_long = (
            (dataframe["close"] >= dataframe["ema9"])
            if self.require_fast_ema_reclaim
            else (dataframe["close"] > dataframe["open"])
        )
        fast_short = (
            (dataframe["close"] <= dataframe["ema9"])
            if self.require_fast_ema_reclaim
            else (dataframe["close"] < dataframe["open"])
        )

        support_rejection = (
            (dataframe["low"] <= dataframe["support"] + self.sweep_atr * dataframe["atr14"])
            & (dataframe["close"] > dataframe["support"])
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["rsi2"].shift(1) <= self.rsi2_extreme)
            & (dataframe["rsi2"] > dataframe["rsi2"].shift(1))
            & fast_long
        )
        resistance_rejection = (
            (dataframe["high"] >= dataframe["resistance"] - self.sweep_atr * dataframe["atr14"])
            & (dataframe["close"] < dataframe["resistance"])
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["rsi2"].shift(1) >= 100.0 - self.rsi2_extreme)
            & (dataframe["rsi2"] < dataframe["rsi2"].shift(1))
            & fast_short
        )

        trend_long = (
            (dataframe["ema50"] > dataframe["ema200"])
            & (dataframe["ema50_15m"] > dataframe["ema200_15m"])
            & (dataframe["close_15m"] > dataframe["ema200_15m"])
            & (dataframe["adx_15m"] >= self.minimum_adx)
            & (dataframe["rsi_15m"] < 72)
        )
        trend_short = (
            (dataframe["ema50"] < dataframe["ema200"])
            & (dataframe["ema50_15m"] < dataframe["ema200_15m"])
            & (dataframe["close_15m"] < dataframe["ema200_15m"])
            & (dataframe["adx_15m"] >= self.minimum_adx)
            & (dataframe["rsi_15m"] > 28)
        )

        dataframe.loc[
            support_rejection & trend_long & liquid,
            ["enter_long", "enter_tag"],
        ] = (1, "support_sweep_long")
        dataframe.loc[
            resistance_rejection & trend_short & liquid,
            ["enter_short", "enter_tag"],
        ] = (1, "resistance_sweep_short")
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


class AlternativeSRMomentumScalperLoose(AlternativeSRMomentumScalper):
    level_window = 8
    rsi2_extreme = 18.0
    minimum_adx = 8.0
    minimum_relative_volume = 0.35
    require_fast_ema_reclaim = False


class AlternativeSRMomentumScalperStrict(AlternativeSRMomentumScalper):
    level_window = 20
    rsi2_extreme = 5.0
    minimum_adx = 18.0
    minimum_relative_volume = 0.85


class AlternativeSRMomentumScalperMedium(AlternativeSRMomentumScalper):
    level_window = 8
    rsi2_extreme = 15.0
    minimum_adx = 10.0
    minimum_relative_volume = 0.45


class AlternativeSRMomentumScalperRisk6(AlternativeSRMomentumScalper):
    stoploss = -0.06
    minimal_roi = {"0": 0.105}


class AlternativeSRMomentumScalperRisk10(AlternativeSRMomentumScalper):
    stoploss = -0.10
    minimal_roi = {"0": 0.165}


class AlternativeSRMomentumScalperRisk20(AlternativeSRMomentumScalper):
    stoploss = -0.20
    minimal_roi = {"0": 0.315}


class AlternativeSRMomentumScalperMediumRisk10(AlternativeSRMomentumScalperMedium):
    stoploss = -0.10
    minimal_roi = {"0": 0.165}


class AlternativeSRMomentumScalperLooseRisk10(AlternativeSRMomentumScalperLoose):
    stoploss = -0.10
    minimal_roi = {"0": 0.165}


class AlternativeSRMomentumScalperStrictRisk10(AlternativeSRMomentumScalperStrict):
    stoploss = -0.10
    minimal_roi = {"0": 0.165}


class AlternativeSelectiveSRScalper(AlternativeSRMomentumScalperMediumRisk10):
    """Use ETH only for resistance shorts; its support longs were unstable."""

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        if metadata["pair"] == "ETH/USDT:USDT":
            dataframe["enter_long"] = 0
        return dataframe


class AlternativeAlignedSRScalper(AlternativeSRMomentumScalperMediumRisk10):
    """Demand immediate 15m alignment in addition to the broad trend regime."""

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        long_aligned = (
            (dataframe["close_15m"] > dataframe["ema50_15m"])
            & (dataframe["ema50_15m"] > dataframe["ema50_15m"].shift(4))
        )
        short_aligned = (
            (dataframe["close_15m"] < dataframe["ema50_15m"])
            & (dataframe["ema50_15m"] < dataframe["ema50_15m"].shift(4))
        )
        dataframe.loc[~long_aligned, "enter_long"] = 0
        dataframe.loc[~short_aligned, "enter_short"] = 0
        return dataframe


class AlternativeSelectiveAlignedSRScalper(AlternativeAlignedSRScalper):
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        if metadata["pair"] == "ETH/USDT:USDT":
            dataframe["enter_long"] = 0
        return dataframe


class AlternativeBreakRetestSRScalper(AlternativeSRMomentumScalper):
    """Enter a pullback to resistance-turned-support (and the short inverse)."""

    breakout_window = 24
    retest_window = 12
    breakout_atr = 0.05
    breakout_relative_volume = 0.8
    retest_atr_tolerance = 0.20
    maximum_entry_distance_ratio = 0.008

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["atr14"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi14"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_sma20"] = dataframe["volume"].rolling(20).mean()
        dataframe["relative_volume"] = dataframe["volume"] / dataframe["volume_sma20"]

        prior_resistance = (
            dataframe["high"].shift(1).rolling(self.breakout_window).max()
        )
        prior_support = dataframe["low"].shift(1).rolling(self.breakout_window).min()
        bullish_breakout = (
            (dataframe["close"] > prior_resistance + self.breakout_atr * dataframe["atr14"])
            & (dataframe["relative_volume"] >= self.breakout_relative_volume)
            & (dataframe["close"] > dataframe["open"])
        )
        bearish_breakout = (
            (dataframe["close"] < prior_support - self.breakout_atr * dataframe["atr14"])
            & (dataframe["relative_volume"] >= self.breakout_relative_volume)
            & (dataframe["close"] < dataframe["open"])
        )

        # A completed breakout only becomes an actionable level on the next bar.
        row = np.arange(len(dataframe), dtype=float)
        long_born = DataFrame({"v": row}, index=dataframe.index)["v"].where(
            bullish_breakout
        )
        short_born = DataFrame({"v": row}, index=dataframe.index)["v"].where(
            bearish_breakout
        )
        dataframe["long_retest_level"] = prior_resistance.where(bullish_breakout).shift(1).ffill()
        dataframe["short_retest_level"] = prior_support.where(bearish_breakout).shift(1).ffill()
        dataframe["long_retest_age"] = row - long_born.shift(1).ffill()
        dataframe["short_retest_age"] = row - short_born.shift(1).ffill()

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
        liquid = (dataframe["volume"] > 0) & (dataframe["relative_volume"] >= 0.35)
        long_retest = (
            dataframe["long_retest_age"].between(1, self.retest_window)
            & (
                dataframe["low"]
                <= dataframe["long_retest_level"] + self.retest_atr_tolerance * dataframe["atr14"]
            )
            & (dataframe["close"] > dataframe["long_retest_level"])
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["close"] > dataframe["ema20"])
            & (dataframe["rsi14"] > 48)
        )
        short_retest = (
            dataframe["short_retest_age"].between(1, self.retest_window)
            & (
                dataframe["high"]
                >= dataframe["short_retest_level"] - self.retest_atr_tolerance * dataframe["atr14"]
            )
            & (dataframe["close"] < dataframe["short_retest_level"])
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["close"] < dataframe["ema20"])
            & (dataframe["rsi14"] < 52)
        )
        # Avoid placing repeated orders for the same retest sequence.
        long_retest &= ~long_retest.shift(1).fillna(False)
        short_retest &= ~short_retest.shift(1).fillna(False)

        trend_long = (
            (dataframe["ema50"] > dataframe["ema200"])
            & (dataframe["ema50_15m"] > dataframe["ema200_15m"])
            & (dataframe["close_15m"] > dataframe["ema200_15m"])
            & (dataframe["adx_15m"] >= self.minimum_adx)
        )
        trend_short = (
            (dataframe["ema50"] < dataframe["ema200"])
            & (dataframe["ema50_15m"] < dataframe["ema200_15m"])
            & (dataframe["close_15m"] < dataframe["ema200_15m"])
            & (dataframe["adx_15m"] >= self.minimum_adx)
        )
        dataframe.loc[long_retest & trend_long & liquid, ["enter_long", "enter_tag"]] = (
            1,
            "breakout_support_retest_long",
        )
        dataframe.loc[short_retest & trend_short & liquid, ["enter_short", "enter_tag"]] = (
            1,
            "breakdown_resistance_retest_short",
        )
        return dataframe

    def custom_entry_price(
        self,
        pair: str,
        trade: Trade | None,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        if not self.dp:
            return proposed_rate
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_rate
        last = dataframe.iloc[-1]
        column = "long_retest_level" if side == "long" else "short_retest_level"
        level = float(last[column])
        if not np.isfinite(level) or level <= 0:
            return proposed_rate
        lower = proposed_rate * (1.0 - self.maximum_entry_distance_ratio)
        upper = proposed_rate * (1.0 + self.maximum_entry_distance_ratio)
        return min(max(level, lower), upper)


class AlternativeBreakRetestSRScalperWide(AlternativeBreakRetestSRScalper):
    breakout_window = 48
    retest_window = 18
    breakout_relative_volume = 1.0
    minimum_adx = 10.0


class AlternativeBreakRetestSRScalperFast(AlternativeBreakRetestSRScalper):
    breakout_window = 12
    retest_window = 8
    breakout_relative_volume = 0.65
    minimum_adx = 8.0
