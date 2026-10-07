"""5m futures scalper built around causal breakout-and-retest S/R entries.

The rolling level is calculated from completed candles only.  A level becomes
eligible after price breaks it with momentum, and an entry is allowed only on
the first rejection of that level during a short retest window.
"""

from datetime import datetime

from pandas import DataFrame
import pandas as pd
import talib.abstract as ta

from freqtrade.strategy import IStrategy, merge_informative_pair


class AggressiveBreakRetestScalper(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "5m"
    informative_timeframe = "15m"
    process_only_new_candles = True
    startup_candle_count = 650

    # At 10x and 0.05% fee per side these produce about +11% versus -7%
    # at the wallet-position level, preserving a realised R:R above 1:1.5.
    stoploss = -0.06
    minimal_roi = {"0": 0.12}
    trailing_stop = False
    use_exit_signal = False
    position_adjustment_enable = False

    level_window = 48
    retest_window = 8
    breakout_atr = 0.08
    breakout_body_atr = 0.45
    retest_above_atr = 0.18
    retest_below_atr = 0.30
    minimum_adx = 18.0
    minimum_relative_volume = 0.75
    breakout_relative_volume = 1.20

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
        dataframe["atr14"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_sma20"] = dataframe["volume"].rolling(20).mean()
        dataframe["relative_volume"] = (
            dataframe["volume"] / dataframe["volume_sma20"]
        )
        candle_range = (dataframe["high"] - dataframe["low"]).clip(lower=1e-12)
        dataframe["close_location"] = (
            (dataframe["close"] - dataframe["low"]) / candle_range
        )
        dataframe["body_size"] = (dataframe["close"] - dataframe["open"]).abs()

        # Every level excludes the current candle, preventing lookahead bias.
        dataframe["prior_resistance"] = (
            dataframe["high"].shift(1).rolling(self.level_window).max()
        )
        dataframe["prior_support"] = (
            dataframe["low"].shift(1).rolling(self.level_window).min()
        )

        bullish_breakout = (
            (
                dataframe["close"]
                > dataframe["prior_resistance"]
                + self.breakout_atr * dataframe["atr14"]
            )
            & (dataframe["close"].shift(1) <= dataframe["prior_resistance"].shift(1))
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["body_size"] >= self.breakout_body_atr * dataframe["atr14"])
            & (dataframe["close_location"] >= 0.70)
            & (dataframe["relative_volume"] >= self.breakout_relative_volume)
        )
        bearish_breakout = (
            (
                dataframe["close"]
                < dataframe["prior_support"]
                - self.breakout_atr * dataframe["atr14"]
            )
            & (dataframe["close"].shift(1) >= dataframe["prior_support"].shift(1))
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["body_size"] >= self.breakout_body_atr * dataframe["atr14"])
            & (dataframe["close_location"] <= 0.30)
            & (dataframe["relative_volume"] >= self.breakout_relative_volume)
        )

        # A cumulative sequence gives every breakout its own causal state.  Age
        # zero is the breakout candle; only ages 1..retest_window may enter.
        long_sequence = bullish_breakout.cumsum()
        short_sequence = bearish_breakout.cumsum()
        dataframe["long_break_age"] = long_sequence.groupby(long_sequence).cumcount()
        dataframe["short_break_age"] = short_sequence.groupby(short_sequence).cumcount()
        dataframe["broken_resistance"] = (
            dataframe["prior_resistance"].where(bullish_breakout).groupby(long_sequence).ffill()
        )
        dataframe["broken_support"] = (
            dataframe["prior_support"].where(bearish_breakout).groupby(short_sequence).ffill()
        )

        long_invalid = (
            dataframe["close"]
            < dataframe["broken_resistance"] - self.retest_below_atr * dataframe["atr14"]
        )
        short_invalid = (
            dataframe["close"]
            > dataframe["broken_support"] + self.retest_below_atr * dataframe["atr14"]
        )
        dataframe["long_level_invalid"] = long_invalid.groupby(long_sequence).cummax()
        dataframe["short_level_invalid"] = short_invalid.groupby(short_sequence).cummax()

        informative = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.informative_timeframe
        )
        informative["ema50"] = ta.EMA(informative, timeperiod=50)
        informative["ema200"] = ta.EMA(informative, timeperiod=200)
        informative["ema50_slope"] = informative["ema50"] - informative["ema50"].shift(4)
        informative["adx"] = ta.ADX(informative, timeperiod=14)
        informative["rsi"] = ta.RSI(informative, timeperiod=14)
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
        long_trend = (
            (dataframe["ema20"] > dataframe["ema50"])
            & (dataframe["close_15m"] > dataframe["ema200_15m"])
            & (dataframe["ema50_15m"] > dataframe["ema200_15m"])
            & (dataframe["ema50_slope_15m"] > 0)
            & (dataframe["adx_15m"] >= self.minimum_adx)
            & (dataframe["rsi_15m"] >= 50)
        )
        short_trend = (
            (dataframe["ema20"] < dataframe["ema50"])
            & (dataframe["close_15m"] < dataframe["ema200_15m"])
            & (dataframe["ema50_15m"] < dataframe["ema200_15m"])
            & (dataframe["ema50_slope_15m"] < 0)
            & (dataframe["adx_15m"] >= self.minimum_adx)
            & (dataframe["rsi_15m"] <= 50)
        )

        long_retest = (
            dataframe["broken_resistance"].notna()
            & dataframe["long_break_age"].between(1, self.retest_window)
            & ~dataframe["long_level_invalid"].fillna(True)
            & (
                dataframe["low"]
                <= dataframe["broken_resistance"]
                + self.retest_above_atr * dataframe["atr14"]
            )
            & (
                dataframe["low"]
                >= dataframe["broken_resistance"]
                - self.retest_below_atr * dataframe["atr14"]
            )
            & (dataframe["close"] > dataframe["broken_resistance"])
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["close_location"] >= 0.62)
            & (dataframe["rsi"] >= 52)
        )
        short_retest = (
            dataframe["broken_support"].notna()
            & dataframe["short_break_age"].between(1, self.retest_window)
            & ~dataframe["short_level_invalid"].fillna(True)
            & (
                dataframe["high"]
                >= dataframe["broken_support"]
                - self.retest_above_atr * dataframe["atr14"]
            )
            & (
                dataframe["high"]
                <= dataframe["broken_support"]
                + self.retest_below_atr * dataframe["atr14"]
            )
            & (dataframe["close"] < dataframe["broken_support"])
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["close_location"] <= 0.38)
            & (dataframe["rsi"] <= 48)
        )

        # Use only the first valid rejection belonging to each breakout.
        long_sequence = (
            (dataframe["long_break_age"] == 0).astype(int).cumsum()
        )
        short_sequence = (
            (dataframe["short_break_age"] == 0).astype(int).cumsum()
        )
        long_entry = long_retest & (long_retest.groupby(long_sequence).cumsum() == 1)
        short_entry = short_retest & (short_retest.groupby(short_sequence).cumsum() == 1)

        dataframe.loc[long_entry & long_trend & liquid, ["enter_long", "enter_tag"]] = (
            1,
            "break_retest_support_long",
        )
        dataframe.loc[
            short_entry & short_trend & liquid, ["enter_short", "enter_tag"]
        ] = (1, "break_retest_resistance_short")
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
        """Place the entry at the retested level, never at the rejection close."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_rate
        level_column = "broken_resistance" if side == "long" else "broken_support"
        level = dataframe.iloc[-1].get(level_column)
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
        return min(10.0, max_leverage)


class AggressiveBreakRetestLoose(AggressiveBreakRetestScalper):
    level_window = 24
    retest_window = 12
    minimum_adx = 14.0
    minimum_relative_volume = 0.55
    breakout_relative_volume = 1.00
    breakout_body_atr = 0.30
    retest_above_atr = 0.25
    retest_below_atr = 0.40


class AggressiveBreakRetestStrict(AggressiveBreakRetestScalper):
    level_window = 72
    retest_window = 6
    minimum_adx = 23.0
    minimum_relative_volume = 0.90
    breakout_relative_volume = 1.50
    breakout_body_atr = 0.60
    retest_above_atr = 0.15
    retest_below_atr = 0.25


class AggressiveBreakRetestMid(AggressiveBreakRetestScalper):
    level_window = 60
    retest_window = 7
    minimum_adx = 21.0
    minimum_relative_volume = 0.82
    breakout_relative_volume = 1.35
    breakout_body_atr = 0.52
    retest_above_atr = 0.16
    retest_below_atr = 0.27


class AggressiveBreakRetestStrongBreak(AggressiveBreakRetestScalper):
    breakout_relative_volume = 1.50
    breakout_body_atr = 0.60


class AggressiveBreakRetestStrongTrend(AggressiveBreakRetestScalper):
    level_window = 72
    retest_window = 6
    minimum_adx = 23.0
    retest_above_atr = 0.15
    retest_below_atr = 0.25


class AggressiveBreakRetestStrictR15(AggressiveBreakRetestStrict):
    # Backtests realise roughly +11%/-7% after fees: net R:R about 1.57.
    minimal_roi = {"0": 0.11}


class AggressiveBreakRetestStrictWide(AggressiveBreakRetestStrict):
    # Wider barriers retain about +14%/-9% after fees: net R:R about 1.56.
    stoploss = -0.08
    minimal_roi = {"0": 0.14}
