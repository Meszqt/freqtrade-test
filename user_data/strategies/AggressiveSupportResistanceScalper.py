"""Aggressive 5m support/resistance rejection scalper for BTC/ETH futures."""

from datetime import datetime, timedelta

from pandas import DataFrame
import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy


class AggressiveSupportResistanceScalper(IStrategy):
    INTERFACE_VERSION = 3

    can_short = True
    timeframe = "5m"
    process_only_new_candles = True
    startup_candle_count = 260

    # At 10x these represent approximately 0.6% adverse price movement and
    # 0.9% favorable price movement: nominal risk:reward = 1:1.5.
    stoploss = -0.06
    # A 11% leveraged net target versus roughly 7% realized stop (including
    # round-trip fees) preserves an effective R:R of about 1:1.57.
    minimal_roi = {"0": 0.11}
    trailing_stop = False
    use_exit_signal = True
    position_adjustment_enable = False

    support_window = 48
    touch_atr_tolerance = 0.18
    wick_body_ratio = 0.75
    minimum_relative_volume = 0.65
    maximum_entry_distance_ratio = 0.012

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "ema50": {"color": "dodgerblue"},
            "ema200": {"color": "white"},
            "support": {"color": "lime"},
            "resistance": {"color": "red"},
        },
        "subplots": {
            "RSI": {"rsi": {"color": "purple"}},
            "Relative volume": {"relative_volume": {"color": "orange"}},
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["atr14"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_sma20"] = dataframe["volume"].rolling(20).mean()
        dataframe["relative_volume"] = dataframe["volume"] / dataframe["volume_sma20"]

        # Shift before rolling: the level for candle i is built exclusively
        # from candles that were already closed before candle i began.
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
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        liquid = (
            (dataframe["volume"] > 0)
            & (dataframe["relative_volume"] >= self.minimum_relative_volume)
        )
        support_rejection = (
            (dataframe["low"] <= dataframe["support"] + self.touch_atr_tolerance * dataframe["atr14"])
            & (dataframe["close"] > dataframe["support"])
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["lower_wick_ratio"] >= self.wick_body_ratio)
        )
        resistance_rejection = (
            (dataframe["high"] >= dataframe["resistance"] - self.touch_atr_tolerance * dataframe["atr14"])
            & (dataframe["close"] < dataframe["resistance"])
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["upper_wick_ratio"] >= self.wick_body_ratio)
        )

        long_condition = (
            support_rejection
            & (dataframe["close"] > dataframe["ema200"])
            & (dataframe["ema50"] >= dataframe["ema50"].shift(3))
            & (dataframe["rsi"].between(28, 52))
            & liquid
        )
        short_condition = (
            resistance_rejection
            & (dataframe["close"] < dataframe["ema200"])
            & (dataframe["ema50"] <= dataframe["ema50"].shift(3))
            & (dataframe["rsi"].between(48, 72))
            & liquid
        )

        dataframe.loc[long_condition, ["enter_long", "enter_tag"]] = (
            1,
            "support_rejection_long",
        )
        dataframe.loc[short_condition, ["enter_short", "enter_tag"]] = (
            1,
            "resistance_rejection_short",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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
        level = float(last["support"] if side == "long" else last["resistance"])
        if level <= 0:
            return proposed_rate
        lower = proposed_rate * (1 - self.maximum_entry_distance_ratio)
        upper = proposed_rate * (1 + self.maximum_entry_distance_ratio)
        return min(max(level, lower), upper)

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | None:
        age = current_time - trade.open_date_utc
        if age >= timedelta(minutes=45) and current_profit > 0.01:
            return "fast_profit_rotation"
        if age >= timedelta(minutes=90):
            return "scalp_time_limit"
        return None

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


class AggressiveSRScalper24(AggressiveSupportResistanceScalper):
    support_window = 24


class AggressiveSRScalper96(AggressiveSupportResistanceScalper):
    support_window = 96


class AggressiveSRScalper12Hold(AggressiveSupportResistanceScalper):
    support_window = 12
    use_exit_signal = False


class AggressiveSRScalper24Hold(AggressiveSupportResistanceScalper):
    support_window = 24
    use_exit_signal = False


class AggressiveSRScalper12MarketHold(AggressiveSRScalper12Hold):
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
        return proposed_rate


class AggressiveSRScalper24MarketHold(AggressiveSRScalper24Hold):
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
        return proposed_rate
