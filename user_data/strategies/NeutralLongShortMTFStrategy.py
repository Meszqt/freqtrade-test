"""Neutral long/short strategy using 15m entries and a 1d trend filter."""

from datetime import datetime
import logging
import os

import numpy as np
import pandas as pd
from pandas import DataFrame
import requests
import talib.abstract as ta

from freqtrade.strategy import IStrategy, merge_informative_pair
from technical import qtpylib


logger = logging.getLogger(__name__)


class NeutralLongShortMTFStrategy(IStrategy):
    INTERFACE_VERSION = 3

    can_short = True
    timeframe = "15m"
    informative_timeframe = "1d"
    process_only_new_candles = True
    startup_candle_count = 400

    # Risk is measured against stake after leverage. A 20% maximum loss and
    # 30% take-profit produce a nominal 1:1.5 risk-to-reward ratio.
    risk_per_trade = 0.20
    reward_per_trade = 0.30
    risk_reward_ratio = 1.5
    orderblock_valid_candles = 96
    minimum_futures_quote_volume = {
        "BTC/USDT:USDT": 100_000_000.0,
        "ETH/USDT:USDT": 50_000_000.0,
    }

    # USDT dominance is calculated as USDT market cap / total crypto market cap.
    # Require both 1h and 1d to move in the same direction before allowing entry.
    cmc_base_url = "https://pro-api.coinmarketcap.com"
    cmc_usdt_id = "825"
    dominance_refresh_seconds = 3600
    dominance_max_age_seconds = 5400
    dominance_min_change_ratio = 0.0001

    # Disable fixed ROI. Once the former +30% target is touched, a 10%
    # positive trailing stop manages the exit instead.
    minimal_roi = {}
    stoploss = -risk_per_trade
    trailing_stop = True
    trailing_stop_positive = 0.10
    trailing_stop_positive_offset = reward_per_trade
    trailing_only_offset_is_reached = True

    # Indicator exits and fixed ROI are disabled. Trades close through the
    # initial stoploss or the positive trailing stop after its offset is reached.
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    position_adjustment_enable = False

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "ema13": {"color": "dodgerblue"},
            "ema21": {"color": "orange"},
            "ema200": {"color": "white"},
        },
        "subplots": {
            "RSI": {"rsi": {"color": "purple"}},
            "Stochastic": {
                "stoch_k": {"color": "blue"},
                "stoch_d": {"color": "red"},
            },
            "Relative volume": {"relative_volume": {"color": "green"}},
        },
    }

    def bot_start(self, **kwargs) -> None:
        self.cmc_api_key = os.getenv("CMC_API_KEY", "").strip()
        self._dominance_last_attempt = None
        self.usdt_dominance_state = {
            "ready": False,
            "updated_at": None,
            "dominance_1h": None,
            "change_1h": None,
            "dominance_1d": None,
            "change_1d": None,
            "allow_long": False,
            "allow_short": False,
        }
        if not self.cmc_api_key and self.config["runmode"].value in ("live", "dry_run"):
            logger.error("CMC_API_KEY is missing; all entries will remain blocked.")

    def _fetch_usdt_dominance(self, interval: str) -> tuple[float, float]:
        headers = {"X-CMC_PRO_API_KEY": self.cmc_api_key, "Accept": "application/json"}
        common_params = {"count": 3, "interval": interval, "convert": "USD"}

        usdt_response = requests.get(
            f"{self.cmc_base_url}/v3/cryptocurrency/quotes/historical",
            headers=headers,
            params={**common_params, "id": self.cmc_usdt_id, "aux": "market_cap"},
            timeout=10,
        )
        usdt_response.raise_for_status()

        global_response = requests.get(
            f"{self.cmc_base_url}/v1/global-metrics/quotes/historical",
            headers=headers,
            params=common_params,
            timeout=10,
        )
        global_response.raise_for_status()

        usdt_quotes = sorted(
            usdt_response.json()["data"][self.cmc_usdt_id]["quotes"],
            key=lambda item: item["timestamp"],
        )
        global_quotes = sorted(
            global_response.json()["data"]["quotes"],
            key=lambda item: item["timestamp"],
        )
        if len(usdt_quotes) < 2 or len(global_quotes) < 2:
            raise ValueError("Not enough USDT dominance history returned by CoinMarketCap")

        previous = (
            float(usdt_quotes[-2]["quote"]["USD"]["market_cap"])
            / float(global_quotes[-2]["quote"]["USD"]["total_market_cap"])
            * 100.0
        )
        current = (
            float(usdt_quotes[-1]["quote"]["USD"]["market_cap"])
            / float(global_quotes[-1]["quote"]["USD"]["total_market_cap"])
            * 100.0
        )
        if previous <= 0 or current <= 0:
            raise ValueError("Invalid USDT dominance value returned by CoinMarketCap")
        return current, (current - previous) / previous

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        if self.config["runmode"].value not in ("live", "dry_run"):
            return
        if not self.cmc_api_key:
            return
        if self._dominance_last_attempt is not None:
            elapsed = (current_time - self._dominance_last_attempt).total_seconds()
            if elapsed < self.dominance_refresh_seconds:
                return

        self._dominance_last_attempt = current_time
        try:
            dominance_1h, change_1h = self._fetch_usdt_dominance("1h")
            dominance_1d, change_1d = self._fetch_usdt_dominance("1d")
            threshold = self.dominance_min_change_ratio

            self.usdt_dominance_state = {
                "ready": True,
                "updated_at": current_time,
                "dominance_1h": dominance_1h,
                "change_1h": change_1h,
                "dominance_1d": dominance_1d,
                "change_1d": change_1d,
                "allow_long": change_1h < -threshold and change_1d < -threshold,
                "allow_short": change_1h > threshold and change_1d > threshold,
            }
            logger.info(
                "USDT dominance updated: 1h=%.4f%% (%+.4f%%), 1d=%.4f%% (%+.4f%%), "
                "long=%s, short=%s",
                dominance_1h,
                change_1h * 100.0,
                dominance_1d,
                change_1d * 100.0,
                self.usdt_dominance_state["allow_long"],
                self.usdt_dominance_state["allow_short"],
            )
        except (requests.RequestException, KeyError, TypeError, ValueError, IndexError) as exc:
            self.usdt_dominance_state["ready"] = False
            self.usdt_dominance_state["allow_long"] = False
            self.usdt_dominance_state["allow_short"] = False
            logger.warning("USDT dominance refresh failed; entries blocked: %s", exc)

    def _add_indicators(self, dataframe: DataFrame) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

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

        dataframe["ema13"] = ta.EMA(dataframe, timeperiod=13)
        dataframe["ema21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)

        # Causal RSI divergences: compare the current extreme only with candles
        # that are already closed 5-20 bars in the past. No future pivot is used.
        bullish_divergence = pd.Series(False, index=dataframe.index)
        bearish_divergence = pd.Series(False, index=dataframe.index)
        for lookback in range(5, 21):
            bullish_divergence |= (
                (dataframe["low"] < dataframe["low"].shift(lookback))
                & (dataframe["rsi"] > dataframe["rsi"].shift(lookback))
                & (dataframe["rsi"] < 30)
            )
            bearish_divergence |= (
                (dataframe["high"] > dataframe["high"].shift(lookback))
                & (dataframe["rsi"] < dataframe["rsi"].shift(lookback))
                & (dataframe["rsi"] > 70)
            )

        dataframe["bullish_divergence"] = bullish_divergence.astype(int)
        dataframe["bearish_divergence"] = bearish_divergence.astype(int)
        dataframe["bullish_divergence_recent"] = (
            dataframe["bullish_divergence"].rolling(5).max().fillna(0)
        )
        dataframe["bearish_divergence_recent"] = (
            dataframe["bearish_divergence"].rolling(5).max().fillna(0)
        )

        # Volume confirmation and deterministic candle-based order blocks.
        dataframe["volume_sma20"] = dataframe["volume"].rolling(20).mean()
        dataframe["relative_volume"] = dataframe["volume"] / dataframe["volume_sma20"]
        dataframe["atr14"] = ta.ATR(dataframe, timeperiod=14)

        body = (dataframe["close"] - dataframe["open"]).abs()
        bullish_displacement = (
            (dataframe["close"] > dataframe["open"])
            & (body > dataframe["atr14"])
            & (dataframe["close"] > dataframe["high"].shift(1))
            & (dataframe["relative_volume"] > 1.5)
        )
        bearish_displacement = (
            (dataframe["close"] < dataframe["open"])
            & (body > dataframe["atr14"])
            & (dataframe["close"] < dataframe["low"].shift(1))
            & (dataframe["relative_volume"] > 1.5)
        )

        bullish_orderblock = bullish_displacement & (
            dataframe["close"].shift(1) < dataframe["open"].shift(1)
        )
        bearish_orderblock = bearish_displacement & (
            dataframe["close"].shift(1) > dataframe["open"].shift(1)
        )

        bullish_low = DataFrame(
            {"value": np.where(bullish_orderblock, dataframe["low"].shift(1), np.nan)},
            index=dataframe.index,
        )["value"]
        bullish_high = DataFrame(
            {"value": np.where(bullish_orderblock, dataframe["open"].shift(1), np.nan)},
            index=dataframe.index,
        )["value"]
        bearish_low = DataFrame(
            {"value": np.where(bearish_orderblock, dataframe["open"].shift(1), np.nan)},
            index=dataframe.index,
        )["value"]
        bearish_high = DataFrame(
            {"value": np.where(bearish_orderblock, dataframe["high"].shift(1), np.nan)},
            index=dataframe.index,
        )["value"]

        # Shift once so a newly detected block is only tradable from the next
        # completed candle onward, avoiding same-candle lookahead behavior.
        dataframe["bullish_ob_low"] = bullish_low.shift(1).ffill(
            limit=self.orderblock_valid_candles
        )
        dataframe["bullish_ob_high"] = bullish_high.shift(1).ffill(
            limit=self.orderblock_valid_candles
        )
        dataframe["bearish_ob_low"] = bearish_low.shift(1).ffill(
            limit=self.orderblock_valid_candles
        )
        dataframe["bearish_ob_high"] = bearish_high.shift(1).ffill(
            limit=self.orderblock_valid_candles
        )
        return dataframe

    def informative_pairs(self):
        if not self.dp:
            return []
        return [(pair, self.informative_timeframe) for pair in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self._add_indicators(dataframe)

        informative = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.informative_timeframe
        )
        informative = self._add_indicators(informative)

        return merge_informative_pair(
            dataframe,
            informative,
            self.timeframe,
            self.informative_timeframe,
            ffill=True,
        )

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dynamic_ema_support = (
            (dataframe["ema13"] > dataframe["ema21"])
            & (dataframe["low"] <= dataframe["ema13"])
            & (dataframe["low"] >= dataframe["ema21"] * 0.995)
            & (dataframe["close"] >= dataframe["ema13"])
            & (dataframe["close"] > dataframe["open"])
        )
        dynamic_ema_resistance = (
            (dataframe["ema13"] < dataframe["ema21"])
            & (dataframe["high"] >= dataframe["ema13"])
            & (dataframe["high"] <= dataframe["ema21"] * 1.005)
            & (dataframe["close"] <= dataframe["ema13"])
            & (dataframe["close"] < dataframe["open"])
        )

        long_condition = (
            (dataframe["close_1d"] > dataframe["ema200_1d"])
            & (dataframe["ema13_1d"] > dataframe["ema21_1d"])
            & (dataframe["rsi_1d"] >= 50)
            & (dataframe["close"] > dataframe["ema200"])
            & (dataframe["bullish_divergence_recent"] > 0)
            & qtpylib.crossed_above(dataframe["rsi"], 30)
            & dynamic_ema_support
            & (dataframe["relative_volume"] >= 1.0)
            & (dataframe["volume"] > 0)
        )

        short_condition = (
            (dataframe["close_1d"] < dataframe["ema200_1d"])
            & (dataframe["ema13_1d"] < dataframe["ema21_1d"])
            & (dataframe["rsi_1d"] <= 50)
            & (dataframe["close"] < dataframe["ema200"])
            & (dataframe["bearish_divergence_recent"] > 0)
            & qtpylib.crossed_below(dataframe["rsi"], 70)
            & dynamic_ema_resistance
            & (dataframe["relative_volume"] >= 1.0)
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[long_condition, ["enter_long", "enter_tag"]] = (
            1,
            "long_rsi_bull_div_ema_support",
        )
        dataframe.loc[short_condition, ["enter_short", "enter_tag"]] = (
            1,
            "short_rsi_bear_div_ema_resistance",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_exit = (
            qtpylib.crossed_below(dataframe["ema13"], dataframe["ema21"])
            | qtpylib.crossed_below(dataframe["rsi"], 45)
            | (
                qtpylib.crossed_below(dataframe["stoch_k"], dataframe["stoch_d"])
                & (dataframe["stoch_k"] > 60)
            )
            | (dataframe["ema13_1d"] < dataframe["ema21_1d"])
        ) & (dataframe["volume"] > 0)

        short_exit = (
            qtpylib.crossed_above(dataframe["ema13"], dataframe["ema21"])
            | qtpylib.crossed_above(dataframe["rsi"], 55)
            | (
                qtpylib.crossed_above(dataframe["stoch_k"], dataframe["stoch_d"])
                & (dataframe["stoch_k"] < 40)
            )
            | (dataframe["ema13_1d"] > dataframe["ema21_1d"])
        ) & (dataframe["volume"] > 0)

        dataframe.loc[long_exit, ["exit_long", "exit_tag"]] = (1, "long_reversal")
        dataframe.loc[short_exit, ["exit_short", "exit_tag"]] = (1, "short_reversal")
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
        """Use pair-specific leverage, capped by the exchange market limit."""
        if pair == "BTC/USDT:USDT":
            return min(100.0, max_leverage)
        if pair == "ETH/USDT:USDT":
            return min(75.0, max_leverage)
        return 1.0

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:
        """Confirm live/dry entries using USDT dominance and futures volume."""
        if not self.dp or self.dp.runmode.value not in ("live", "dry_run"):
            return True

        try:
            dominance = getattr(self, "usdt_dominance_state", {"ready": False})
            updated_at = dominance.get("updated_at")
            if not dominance.get("ready") or updated_at is None:
                return False
            if (current_time - updated_at).total_seconds() > self.dominance_max_age_seconds:
                return False
            if side == "long" and not dominance.get("allow_long", False):
                return False
            if side == "short" and not dominance.get("allow_short", False):
                return False

            ticker = self.dp.ticker(pair)
            quote_volume = float(ticker.get("quoteVolume", 0.0))
            required_quote_volume = self.minimum_futures_quote_volume.get(pair, float("inf"))
            if quote_volume < required_quote_volume:
                return False
            return side in ("long", "short")
        except Exception:
            return False
