"""Convert the EMA200 reaction event study into a conservative equity simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TIMEFRAME_ORDER = ["1h", "2h", "4h", "8h", "12h", "1d"]


def simulate(
    trades: pd.DataFrame,
    starting_equity: float,
    stake: float,
    leverage: float,
    fee_rate: float,
) -> tuple[dict, pd.DataFrame]:
    trades = trades.sort_values(["simulation_exit_date", "entry_date"]).copy()
    direction_return = np.where(
        trades["direction"].eq("support"),
        trades["simulation_exit_price"] / trades["entry_price"] - 1,
        1 - trades["simulation_exit_price"] / trades["entry_price"],
    )
    fee_return = fee_rate * (
        1 + trades["simulation_exit_price"] / trades["entry_price"]
    )
    trades["gross_return_pct"] = 100 * leverage * direction_return
    trades["fee_usdt"] = stake * leverage * fee_return
    trades["net_profit_usdt"] = stake * leverage * (direction_return - fee_return)
    trades["net_return_on_stake_pct"] = 100 * trades["net_profit_usdt"] / stake
    trades["equity"] = starting_equity + trades["net_profit_usdt"].cumsum()
    trades["equity_peak"] = trades["equity"].cummax().clip(lower=starting_equity)
    trades["drawdown_pct"] = (
        100 * (trades["equity_peak"] - trades["equity"]) / trades["equity_peak"]
    )

    wins = trades.loc[trades["net_profit_usdt"] > 0, "net_profit_usdt"].sum()
    losses = -trades.loc[trades["net_profit_usdt"] < 0, "net_profit_usdt"].sum()
    net_profit = trades["net_profit_usdt"].sum()
    first_entry = pd.to_datetime(trades["entry_date"], utc=True).min()
    last_exit = pd.to_datetime(trades["simulation_exit_date"], utc=True).max()
    years = max((last_exit - first_entry).total_seconds() / (365.25 * 86400), 1 / 365.25)
    final_equity = starting_equity + net_profit
    cagr = (
        100 * ((final_equity / starting_equity) ** (1 / years) - 1)
        if final_equity > 0
        else -100.0
    )
    summary = {
        "trades": len(trades),
        "targets": int(trades["simulation_exit_reason"].eq("target").sum()),
        "stops": int(trades["simulation_exit_reason"].eq("stop").sum()),
        "time_exits": int(trades["simulation_exit_reason"].eq("time_exit").sum()),
        "win_rate_pct": 100 * trades["net_profit_usdt"].gt(0).mean(),
        "gross_profit_before_fees_usdt": float(
            (stake * leverage * direction_return).sum()
        ),
        "fees_usdt": float(trades["fee_usdt"].sum()),
        "net_profit_usdt": float(net_profit),
        "total_roi_pct": 100 * net_profit / starting_equity,
        "final_equity_usdt": float(final_equity),
        "cagr_pct": float(cagr),
        "profit_factor": float(wins / losses) if losses else float("inf"),
        "max_drawdown_pct": float(trades["drawdown_pct"].max()),
        "average_net_return_on_stake_pct": float(
            trades["net_return_on_stake_pct"].mean()
        ),
        "first_entry": first_entry,
        "last_exit": last_exit,
    }
    return summary, trades


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("/freqtrade/user_data/backtest_results/ema200_mtf/events.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/freqtrade/user_data/backtest_results/ema200_mtf"),
    )
    parser.add_argument("--starting-equity", type=float, default=1000.0)
    parser.add_argument("--stake", type=float, default=30.0)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    args = parser.parse_args()

    events = pd.read_csv(
        args.events,
        parse_dates=["date", "entry_date", "simulation_exit_date"],
    )
    base = events[
        np.isclose(events["tolerance_pct"], 0.25)
        & np.isclose(events["target_atr"], 1.0)
        & events["trade_eligible"].astype(bool)
    ].copy()

    summaries = []
    details = []
    for timeframe in TIMEFRAME_ORDER:
        scoped = base[base["timeframe"] == timeframe]
        if scoped.empty:
            continue
        summary, simulated = simulate(
            scoped,
            args.starting_equity,
            args.stake,
            args.leverage,
            args.fee_rate,
        )
        summary["timeframe"] = timeframe
        summaries.append(summary)
        simulated["simulation_timeframe"] = timeframe
        details.append(simulated)

    summary_frame = pd.DataFrame(summaries)
    summary_frame["timeframe"] = pd.Categorical(
        summary_frame["timeframe"], categories=TIMEFRAME_ORDER, ordered=True
    )
    summary_frame = summary_frame.sort_values("timeframe")
    detail_frame = pd.concat(details, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(args.output_dir / "equity_summary_1x.csv", index=False)
    detail_frame.to_csv(args.output_dir / "equity_trades_1x.csv", index=False)

    print(
        f"Starting equity={args.starting_equity:.2f} USDT, stake={args.stake:.2f} USDT, "
        f"leverage={args.leverage:g}x, fee={args.fee_rate * 100:.3f}% per side"
    )
    print(
        summary_frame[
            [
                "timeframe",
                "trades",
                "win_rate_pct",
                "net_profit_usdt",
                "total_roi_pct",
                "final_equity_usdt",
                "profit_factor",
                "max_drawdown_pct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
