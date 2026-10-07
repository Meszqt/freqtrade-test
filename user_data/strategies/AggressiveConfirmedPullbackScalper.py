"""5m aggressive scalper: support/resistance pullback followed by momentum confirmation."""

from datetime import datetime

from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy import IStrategy, merge_informative_pair
from technical import qtpylib


class AggressiveConfirmedPullbackScalper(IStrategy):
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

    support_period = 50
    confirmation_period = 20
    touch_memory = 4
    minimum_adx = 18.0
    minimum_relative_volume = 0.65
    confirmation_mode = "ema_cross"

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
        dataframe["ema9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema34"] = ta.EMA(dataframe, timeperiod=34)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["support_level"] = ta.EMA(dataframe, timeperiod=self.support_period)
        dataframe["confirmation_ema"] = ta.EMA(
            dataframe, timeperiod=self.confirmation_period
        )
        dataframe["atr14"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_sma20"] = dataframe["volume"].rolling(20).mean()
        dataframe["relative_volume"] = dataframe["volume"] / dataframe["volume_sma20"]

        touched_long = dataframe["low"] <= dataframe["support_level"] + 0.20 * dataframe["atr14"]
        touched_short = dataframe["high"] >= dataframe["support_level"] - 0.20 * dataframe["atr14"]
        dataframe["long_touch_recent"] = touched_long.rolling(self.touch_memory).max().fillna(0)
        dataframe["short_touch_recent"] = touched_short.rolling(self.touch_memory).max().fillna(0)

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
        if self.confirmation_mode == "candle_break":
            long_confirmation = dataframe["close"] > dataframe["high"].shift(1)
            short_confirmation = dataframe["close"] < dataframe["low"].shift(1)
        else:
            long_confirmation = qtpylib.crossed_above(
                dataframe["close"], dataframe["confirmation_ema"]
            )
            short_confirmation = qtpylib.crossed_below(
                dataframe["close"], dataframe["confirmation_ema"]
            )

        liquid = (
            (dataframe["volume"] > 0)
            & (dataframe["relative_volume"] >= self.minimum_relative_volume)
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
        long_entry = (
            (dataframe["long_touch_recent"] > 0)
            & (dataframe["close"] > dataframe["support_level"])
            & (dataframe["rsi"] > 50)
            & long_confirmation
            & long_trend
            & liquid
        )
        short_entry = (
            (dataframe["short_touch_recent"] > 0)
            & (dataframe["close"] < dataframe["support_level"])
            & (dataframe["rsi"] < 50)
            & short_confirmation
            & short_trend
            & liquid
        )
        dataframe.loc[long_entry, ["enter_long", "enter_tag"]] = (
            1,
            "confirmed_support_long",
        )
        dataframe.loc[short_entry, ["enter_short", "enter_tag"]] = (
            1,
            "confirmed_resistance_short",
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


class AggressiveConfirmedEMA9From34(AggressiveConfirmedPullbackScalper):
    support_period = 34
    confirmation_period = 9


class AggressiveConfirmedCandleBreak(AggressiveConfirmedPullbackScalper):
    support_period = 34
    confirmation_mode = "candle_break"
