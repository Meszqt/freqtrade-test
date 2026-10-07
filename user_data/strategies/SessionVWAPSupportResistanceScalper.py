"""Causal 5 minute VWAP-band support/resistance scalper.

The strategy treats the lower VWAP volatility band as dynamic support and the
upper band as dynamic resistance.  It enters only after a completed 5 minute
candle rejects a band, while the broader 5 minute trend points in the same
direction.  Every value used by the signal is available at candle close; an
entry is therefore placed no earlier than the next candle.

The small subclasses are deliberately bounded research variants.  They share
the same signal logic and differ only in whether VWAP is anchored to the UTC
session or rolled over 24 hours, and in the band selectivity.
"""

from datetime import datetime

import numpy as np
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy import IStrategy


class SessionVWAPSupportResistanceScalper(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "5m"
    process_only_new_candles = True
    startup_candle_count = 650

    # At 10x leverage and 0.05% fees on entry and exit, a conventional stop
    # completes near -6%, while an ROI winner completes near +9%.  The realised
    # net reward:risk is therefore about 1.5:1 before slippage.
    stoploss = -0.05
    minimal_roi = {"0": 0.09}
    trailing_stop = False
    use_exit_signal = False
    position_adjustment_enable = False

    use_rolling_vwap = False
    rolling_window = 288  # 24 hours on 5 minute candles
    band_sigma = 1.10
    atr_width_floor = 0.80
    maximum_band_width_atr = 2.50
    touch_tolerance_atr = 0.12
    maximum_reclaim_atr = 0.45
    minimum_relative_volume = 0.55
    previous_rsi_long_max = 40.0
    previous_rsi_short_min = 60.0
    minimum_wick_body_ratio = 0.25
    minimum_session_bars = 12
    minimum_band_width_pct = 0.0015
    maximum_band_width_pct = 0.025
    require_adx = 12.0
    require_ema200_alignment = True
    require_vwap_slope = False
    trend_slope_bars = 6

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    @staticmethod
    def _safe_weighted_variance(
        weighted_price: DataFrame,
        weighted_square: DataFrame,
        cumulative_volume: DataFrame,
    ) -> DataFrame:
        mean = weighted_price / cumulative_volume
        variance = weighted_square / cumulative_volume - mean.pow(2)
        return variance.clip(lower=0.0)

    def _populate_vwap(self, dataframe: DataFrame) -> DataFrame:
        typical = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3.0
        pv = typical * dataframe["volume"]
        p2v = typical.pow(2) * dataframe["volume"]

        if self.use_rolling_vwap:
            cumulative_volume = dataframe["volume"].rolling(
                self.rolling_window, min_periods=self.rolling_window
            ).sum()
            weighted_price = pv.rolling(
                self.rolling_window, min_periods=self.rolling_window
            ).sum()
            weighted_square = p2v.rolling(
                self.rolling_window, min_periods=self.rolling_window
            ).sum()
            dataframe["session_bars"] = self.rolling_window
        else:
            # Binance crypto sessions are anchored at 00:00 UTC.  Grouped
            # cumulative sums use only the current and earlier candles.
            session = dataframe["date"].dt.floor("D")
            cumulative_volume = dataframe["volume"].groupby(session).cumsum()
            weighted_price = pv.groupby(session).cumsum()
            weighted_square = p2v.groupby(session).cumsum()
            dataframe["session_bars"] = dataframe.groupby(session).cumcount() + 1

        cumulative_volume = cumulative_volume.replace(0.0, np.nan)
        dataframe["vwap"] = weighted_price / cumulative_volume
        variance = self._safe_weighted_variance(
            weighted_price, weighted_square, cumulative_volume
        )
        dataframe["vwap_sigma"] = np.sqrt(variance)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["atr14"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["adx14"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi7"] = ta.RSI(dataframe, timeperiod=7)
        dataframe["volume_sma20"] = dataframe["volume"].rolling(20).mean()
        dataframe["relative_volume"] = (
            dataframe["volume"] / dataframe["volume_sma20"]
        )

        dataframe = self._populate_vwap(dataframe)
        unconstrained_width = np.maximum(
            self.band_sigma * dataframe["vwap_sigma"],
            self.atr_width_floor * dataframe["atr14"],
        )
        dataframe["band_width"] = np.minimum(
            unconstrained_width,
            self.maximum_band_width_atr * dataframe["atr14"],
        )
        dataframe["vwap_support"] = dataframe["vwap"] - dataframe["band_width"]
        dataframe["vwap_resistance"] = dataframe["vwap"] + dataframe["band_width"]
        dataframe["band_width_pct"] = dataframe["band_width"] / dataframe["close"]

        body = (dataframe["close"] - dataframe["open"]).abs().clip(lower=1e-12)
        dataframe["lower_wick_body"] = (
            dataframe[["open", "close"]].min(axis=1) - dataframe["low"]
        ) / body
        dataframe["upper_wick_body"] = (
            dataframe["high"] - dataframe[["open", "close"]].max(axis=1)
        ) / body
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        valid_market = (
            (dataframe["volume"] > 0)
            & (dataframe["relative_volume"] >= self.minimum_relative_volume)
            & (dataframe["session_bars"] >= self.minimum_session_bars)
            & dataframe["band_width_pct"].between(
                self.minimum_band_width_pct, self.maximum_band_width_pct
            )
            & (dataframe["adx14"] >= self.require_adx)
        )

        support_rejection = (
            (
                dataframe["low"]
                <= dataframe["vwap_support"]
                + self.touch_tolerance_atr * dataframe["atr14"]
            )
            & (dataframe["close"] > dataframe["vwap_support"])
            & (
                dataframe["close"] - dataframe["vwap_support"]
                <= self.maximum_reclaim_atr * dataframe["atr14"]
            )
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["lower_wick_body"] >= self.minimum_wick_body_ratio)
            & (dataframe["rsi7"].shift(1) <= self.previous_rsi_long_max)
            & (dataframe["rsi7"] > dataframe["rsi7"].shift(1))
        )
        resistance_rejection = (
            (
                dataframe["high"]
                >= dataframe["vwap_resistance"]
                - self.touch_tolerance_atr * dataframe["atr14"]
            )
            & (dataframe["close"] < dataframe["vwap_resistance"])
            & (
                dataframe["vwap_resistance"] - dataframe["close"]
                <= self.maximum_reclaim_atr * dataframe["atr14"]
            )
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["upper_wick_body"] >= self.minimum_wick_body_ratio)
            & (dataframe["rsi7"].shift(1) >= self.previous_rsi_short_min)
            & (dataframe["rsi7"] < dataframe["rsi7"].shift(1))
        )

        # Pullbacks are accepted only in the dominant direction.  A positive
        # EMA slope avoids buying a lower band in a fresh breakdown (and the
        # inverse for shorts).
        long_trend = dataframe["ema20"] > dataframe["ema50"]
        short_trend = dataframe["ema20"] < dataframe["ema50"]
        if self.require_ema200_alignment:
            long_trend &= (
                (dataframe["ema50"] > dataframe["ema200"])
                & (dataframe["close"] > dataframe["ema200"])
            )
            short_trend &= (
                (dataframe["ema50"] < dataframe["ema200"])
                & (dataframe["close"] < dataframe["ema200"])
            )
        if self.trend_slope_bars > 0:
            long_trend &= dataframe["ema50"] > dataframe["ema50"].shift(
                self.trend_slope_bars
            )
            short_trend &= dataframe["ema50"] < dataframe["ema50"].shift(
                self.trend_slope_bars
            )
        if self.require_vwap_slope:
            long_trend &= dataframe["vwap"] > dataframe["vwap"].shift(3)
            short_trend &= dataframe["vwap"] < dataframe["vwap"].shift(3)

        dataframe.loc[
            support_rejection & long_trend & valid_market,
            ["enter_long", "enter_tag"],
        ] = (1, "vwap_band_support_long")
        dataframe.loc[
            resistance_rejection & short_trend & valid_market,
            ["enter_short", "enter_tag"],
        ] = (1, "vwap_band_resistance_short")
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


class SessionVWAPSupportResistanceScalperLoose(SessionVWAPSupportResistanceScalper):
    band_sigma = 0.45
    atr_width_floor = 0.55
    maximum_band_width_atr = 1.60
    touch_tolerance_atr = 0.20
    maximum_reclaim_atr = 0.70
    minimum_relative_volume = 0.25
    previous_rsi_long_max = 48.0
    previous_rsi_short_min = 52.0
    minimum_wick_body_ratio = 0.0
    require_adx = 6.0


class SessionVWAPSupportResistanceScalperExtreme(SessionVWAPSupportResistanceScalper):
    band_sigma = 1.45
    atr_width_floor = 1.10
    minimum_relative_volume = 0.70
    previous_rsi_long_max = 34.0
    previous_rsi_short_min = 66.0
    minimum_wick_body_ratio = 0.50
    require_adx = 15.0


class RollingVWAPSupportResistanceScalper(SessionVWAPSupportResistanceScalper):
    use_rolling_vwap = True
    rolling_window = 96
    band_sigma = 0.65
    atr_width_floor = 0.65
    maximum_band_width_atr = 1.80


class RollingVWAPSupportResistanceScalperLoose(RollingVWAPSupportResistanceScalper):
    rolling_window = 48
    band_sigma = 0.45
    atr_width_floor = 0.55
    maximum_band_width_atr = 1.50
    touch_tolerance_atr = 0.20
    maximum_reclaim_atr = 0.70
    minimum_relative_volume = 0.25
    previous_rsi_long_max = 48.0
    previous_rsi_short_min = 52.0
    minimum_wick_body_ratio = 0.0
    require_adx = 6.0


class RollingVWAPSupportResistanceScalperFast(RollingVWAPSupportResistanceScalperLoose):
    """Higher-frequency variant using only the responsive 20/50 EMA regime."""

    require_ema200_alignment = False
    trend_slope_bars = 3
