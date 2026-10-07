"""Backtest causal limit entries placed exactly at the last closed EMA200."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ema200_mtf_reaction_backtest import TIMEFRAMES, load_15m, resample_ohlcv


TIMEFRAME_DELTA = {
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4),
    "8h": pd.Timedelta(hours=8),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1),
}


def simulate_pair_timeframe(
    base: pd.DataFrame,
    candles: pd.DataFrame,
    pair: str,
    timeframe: str,
    target_atr: float,
    stop_atr: float,
    horizon: int,
    stake: float,
    leverage: float,
    fee_rate: float,
) -> list[dict]:
    trades: list[dict] = []
    delta = TIMEFRAME_DELTA[timeframe]
    next_entry_time = candles.index[1]
    ema50 = candles["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    volume_sma20 = candles["volume"].rolling(20).mean()

    for i in range(1, len(candles)):
        candle_time = candles.index[i]
        previous_time = candles.index[i - 1]
        if candle_time < next_entry_time or candle_time - previous_time != delta:
            continue

        # The order for candle i uses only values known after candle i-1 closed.
        level = float(candles["ema200"].iloc[i - 1])
        atr = float(candles["atr14"].iloc[i - 1])
        previous_close = float(candles["close"].iloc[i - 1])
        if previous_close > level:
            direction = "support"
            direction_sign = 1.0
            target = level + target_atr * atr
            stop = level - stop_atr * atr
        elif previous_close < level:
            direction = "resistance"
            direction_sign = -1.0
            target = level - target_atr * atr
            stop = level + stop_atr * atr
        else:
            continue

        current_low = float(candles["low"].iloc[i])
        current_high = float(candles["high"].iloc[i])
        if not (current_low <= level <= current_high):
            continue

        end_time = candle_time + horizon * delta
        detail = base.loc[(base.index >= candle_time) & (base.index < end_time)]
        if detail.empty:
            continue

        filled = False
        entry_time = None
        exit_time = detail.index[-1]
        exit_price = float(detail["close"].iloc[-1])
        exit_reason = "time_exit"

        for detail_time, bar in detail.iterrows():
            if not filled:
                if not (float(bar["low"]) <= level <= float(bar["high"])):
                    continue
                filled = True
                entry_time = detail_time

            if direction == "support":
                hit_stop = float(bar["low"]) <= stop
                hit_target = float(bar["high"]) >= target
            else:
                hit_stop = float(bar["high"]) >= stop
                hit_target = float(bar["low"]) <= target

            # The exact path inside a 15m candle is unknown.  Stop wins ties.
            if hit_stop:
                exit_time = detail_time
                exit_price = stop
                exit_reason = "stop"
                break
            if hit_target:
                exit_time = detail_time
                exit_price = target
                exit_reason = "target"
                break

        if not filled or entry_time is None:
            continue

        direction_return = (
            exit_price / level - 1
            if direction == "support"
            else 1 - exit_price / level
        )
        fee_return = fee_rate * (1 + exit_price / level)
        gross_profit = stake * leverage * direction_return
        fees = stake * leverage * fee_return
        net_profit = gross_profit - fees

        trades.append(
            {
                "pair": pair,
                "timeframe": timeframe,
                "direction": direction,
                "signal_candle": candle_time,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": level,
                "target_price": target,
                "stop_price": stop,
                "exit_price": exit_price,
                "atr14": atr,
                "atr_pct": 100 * atr / level,
                "approach_distance_atr": abs(previous_close - level) / atr,
                "prior_body_aligned_atr": direction_sign
                * (
                    float(candles["close"].iloc[i - 1])
                    - float(candles["open"].iloc[i - 1])
                )
                / atr,
                "ema50_alignment_atr": direction_sign
                * (float(ema50.iloc[i - 1]) - level)
                / atr,
                "relative_volume20": float(candles["volume"].iloc[i - 1])
                / float(volume_sma20.iloc[i - 1]),
                "ema200_slope_5_atr": direction_sign
                * (level - float(candles["ema200"].iloc[max(0, i - 6)]))
                / atr,
                "ema200_slope_10_atr": direction_sign
                * (level - float(candles["ema200"].iloc[max(0, i - 11)]))
                / atr,
                "ema200_slope_20_atr": direction_sign
                * (level - float(candles["ema200"].iloc[max(0, i - 21)]))
                / atr,
                "exit_reason": exit_reason,
                "gross_return_pct": 100 * leverage * direction_return,
                "fees_usdt": fees,
                "net_profit_usdt": net_profit,
                "net_return_on_stake_pct": 100 * net_profit / stake,
            }
        )
        # One position per pair. A fresh order is allowed after this exit.
        next_entry_time = exit_time + pd.Timedelta(minutes=15)

    return trades


def summarize(
    trades: pd.DataFrame, starting_equity: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    equity_frames = []
    for timeframe in TIMEFRAMES:
        scoped = trades[trades["timeframe"] == timeframe].sort_values("exit_time").copy()
        scoped["equity"] = starting_equity + scoped["net_profit_usdt"].cumsum()
        scoped["peak"] = scoped["equity"].cummax().clip(lower=starting_equity)
        scoped["drawdown_pct"] = 100 * (scoped["peak"] - scoped["equity"]) / scoped["peak"]
        wins = scoped.loc[scoped["net_profit_usdt"] > 0, "net_profit_usdt"].sum()
        losses = -scoped.loc[scoped["net_profit_usdt"] < 0, "net_profit_usdt"].sum()
        net_profit = scoped["net_profit_usdt"].sum()
        summaries.append(
            {
                "timeframe": timeframe,
                "trades": len(scoped),
                "targets": int(scoped["exit_reason"].eq("target").sum()),
                "stops": int(scoped["exit_reason"].eq("stop").sum()),
                "time_exits": int(scoped["exit_reason"].eq("time_exit").sum()),
                "ema_respect_target_rate_pct": 100 * scoped["exit_reason"].eq("target").mean(),
                "net_win_rate_pct": 100 * scoped["net_profit_usdt"].gt(0).mean(),
                "gross_profit_before_fees_usdt": scoped["net_profit_usdt"].sum()
                + scoped["fees_usdt"].sum(),
                "fees_usdt": scoped["fees_usdt"].sum(),
                "net_profit_usdt": net_profit,
                "total_roi_pct": 100 * net_profit / starting_equity,
                "final_equity_usdt": starting_equity + net_profit,
                "profit_factor": wins / losses if losses else float("inf"),
                "max_drawdown_pct": scoped["drawdown_pct"].max(),
            }
        )
        equity_frames.append(scoped)
    return pd.DataFrame(summaries), pd.concat(equity_frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/freqtrade/user_data/data/binance/futures"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/freqtrade/user_data/backtest_results/ema200_exact_entry"),
    )
    parser.add_argument("--starting-equity", type=float, default=1000.0)
    parser.add_argument("--stake", type=float, default=30.0)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--target-atr", type=float, default=1.0)
    parser.add_argument("--stop-atr", type=float, default=0.5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--timeframes", nargs="+", choices=list(TIMEFRAMES), default=list(TIMEFRAMES)
    )
    args = parser.parse_args()

    all_trades: list[dict] = []
    for source in sorted(args.data_dir.glob("*-15m-futures.feather")):
        pair = source.name.split("-")[0].replace("_", "/", 1).replace("_", ":")
        base = load_15m(source)
        for timeframe, (rule, expected_bars) in TIMEFRAMES.items():
            if timeframe not in args.timeframes:
                continue
            candles = resample_ohlcv(base, rule, expected_bars)
            all_trades.extend(
                simulate_pair_timeframe(
                    base,
                    candles,
                    pair,
                    timeframe,
                    args.target_atr,
                    args.stop_atr,
                    args.horizon,
                    args.stake,
                    args.leverage,
                    args.fee_rate,
                )
            )

    trades = pd.DataFrame(all_trades)
    summary, trades = summarize(trades, args.starting_equity)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary_1x.csv", index=False)
    trades.to_csv(args.output_dir / "trades_1x.csv", index=False)

    print(
        f"Exact EMA200 entry | equity={args.starting_equity:.2f}, stake={args.stake:.2f}, "
        f"leverage={args.leverage:g}x, fee={args.fee_rate * 100:.3f}%/side, "
        f"target={args.target_atr:g} ATR, stop={args.stop_atr:g} ATR"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
