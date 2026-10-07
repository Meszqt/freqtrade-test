"""5m support/resistance scalper with asymmetric, causal trend regimes."""

from datetime import datetime

from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy import IStrategy, merge_informative_pair


class AggressiveRegimeSRScalper(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "5m"
    informative_timeframe = "15m"
    process_only_new_candles = True
    startup_candle_count = 650

    # With 10x leverage and the configured 0.05% fee per side, completed
    # winners/losses are approximately +11%/-7%: net R:R about 1.57.
    stoploss = -0.06
    minimal_roi = {"0": 0.11}
    trailing_stop = False
    use_exit_signal = False
    position_adjustment_enable = False

    support_window = 12
    touch_atr_tolerance = 0.18
    wick_body_ratio = 0.75
    minimum_relative_volume = 0.65

    long_gap_min = 1.0
    long_gap_max = 2.5
    long_atr_pct_min = 0.003
    long_atr_pct_max = 0.005
    short_gap_min = 1.0
    short_gap_max = 2.0

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
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["atr14"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr14"] / dataframe["close"]
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_sma20"] = dataframe["volume"].rolling(20).mean()
        dataframe["relative_volume"] = (
            dataframe["volume"] / dataframe["volume_sma20"]
        )
        dataframe["support"] = (
            dataframe["low"].shift(1).rolling(self.support_window).min()
        )
        dataframe["resistance"] = (
            dataframe["high"].shift(1).rolling(self.support_window).max()
        )

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
        informative["ema50"] = ta.EMA(informative, timeperiod=50)
        informative["ema200"] = ta.EMA(informative, timeperiod=200)
        informative["atr14"] = ta.ATR(informative, timeperiod=14)
        dataframe = merge_informative_pair(
            dataframe,
            informative,
            self.timeframe,
            self.informative_timeframe,
            ffill=True,
        )
        dataframe["long_gap_atr"] = (
            (dataframe["ema50_15m"] - dataframe["ema200_15m"])
            / dataframe["atr14_15m"]
        )
        dataframe["short_gap_atr"] = -dataframe["long_gap_atr"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        liquid = (
            (dataframe["volume"] > 0)
            & (dataframe["relative_volume"] >= self.minimum_relative_volume)
        )
        support_rejection = (
            (
                dataframe["low"]
                <= dataframe["support"]
                + self.touch_atr_tolerance * dataframe["atr14"]
            )
            & (dataframe["close"] > dataframe["support"])
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["lower_wick_ratio"] >= self.wick_body_ratio)
        )
        resistance_rejection = (
            (
                dataframe["high"]
                >= dataframe["resistance"]
                - self.touch_atr_tolerance * dataframe["atr14"]
            )
            & (dataframe["close"] < dataframe["resistance"])
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["upper_wick_ratio"] >= self.wick_body_ratio)
        )

        long_regime = (
            dataframe["long_gap_atr"].between(self.long_gap_min, self.long_gap_max)
            & dataframe["atr_pct"].between(
                self.long_atr_pct_min, self.long_atr_pct_max
            )
        )
        short_regime = dataframe["short_gap_atr"].between(
            self.short_gap_min, self.short_gap_max
        )
        long_entry = (
            support_rejection
            & (dataframe["close"] > dataframe["ema200"])
            & (dataframe["ema50"] >= dataframe["ema50"].shift(3))
            & dataframe["rsi"].between(28, 52)
            & long_regime
            & liquid
        )
        short_entry = (
            resistance_rejection
            & (dataframe["close"] < dataframe["ema200"])
            & (dataframe["ema50"] <= dataframe["ema50"].shift(3))
            & dataframe["rsi"].between(48, 72)
            & short_regime
            & liquid
        )
        dataframe.loc[long_entry, ["enter_long", "enter_tag"]] = (
            1,
            "support_rejection_long",
        )
        dataframe.loc[short_entry, ["enter_short", "enter_tag"]] = (
            1,
            "resistance_rejection_short",
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


class AggressiveRegimeSRGapOnly(AggressiveRegimeSRScalper):
    long_atr_pct_min = 0.0
    long_atr_pct_max = 1.0


class AggressiveRegimeSRWideLong(AggressiveRegimeSRScalper):
    long_gap_min = 0.75
    long_gap_max = 3.0
    long_atr_pct_min = 0.0025
    long_atr_pct_max = 0.006


class AggressiveRegimeSRTightVolume(AggressiveRegimeSRScalper):
    minimum_relative_volume = 1.0

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        too_much_volume = dataframe["relative_volume"] > 1.25
        dataframe.loc[too_much_volume, ["enter_long", "enter_short"]] = 0
        return dataframe
