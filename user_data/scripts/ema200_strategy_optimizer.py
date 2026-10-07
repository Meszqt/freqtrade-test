"""Select a causal EMA200 pullback filter with development/validation/test splits."""

from __future__ import annotations

import itertools
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path("/freqtrade/user_data/backtest_results/ema200_strategy_research/trades_1x.csv")
OUTPUT = Path("/freqtrade/user_data/backtest_results/ema200_strategy_research")


def stats(frame: pd.DataFrame) -> dict:
    gains = frame.loc[frame["net_profit_usdt"] > 0, "net_profit_usdt"].sum()
    losses = -frame.loc[frame["net_profit_usdt"] < 0, "net_profit_usdt"].sum()
    return {
        "trades": len(frame),
        "net": frame["net_profit_usdt"].sum(),
        "pf": gains / losses if losses else float("inf"),
        "win_rate": 100 * frame["net_profit_usdt"].gt(0).mean() if len(frame) else 0.0,
        "expectancy": frame["net_profit_usdt"].mean() if len(frame) else 0.0,
    }


def apply_candidate(frame: pd.DataFrame, candidate: dict) -> pd.DataFrame:
    return frame[
        (frame["timeframe"] == candidate["timeframe"])
        & (frame["ema50_alignment_atr"] >= candidate["ema50_min"])
        & (frame["ema200_slope_20_atr"] >= candidate["slope20_min"])
        & (frame["approach_distance_atr"] <= candidate["approach_max"])
        & (frame["prior_body_aligned_atr"] <= candidate["prior_body_max"])
        & (frame["relative_volume20"] >= candidate["volume_min"])
    ]


def compounded_portfolio(
    frame: pd.DataFrame,
    starting_equity: float = 1000.0,
    risk_fraction: float = 0.01,
    max_position_fraction: float = 0.50,
) -> dict:
    frame = frame.sort_values(["entry_time", "exit_time"]).copy()
    events = []
    for row_id, row in frame.iterrows():
        events.append((row["entry_time"], 1, row_id))
        events.append((row["exit_time"], 0, row_id))
    # Exits at a timestamp are processed before new entries at that timestamp.
    events.sort(key=lambda item: (item[0], item[1]))

    equity = starting_equity
    peak = starting_equity
    max_drawdown = 0.0
    open_notional: dict[int, float] = {}
    equity_points = []
    for timestamp, event_type, row_id in events:
        row = frame.loc[row_id]
        if event_type == 1:
            stop_distance = abs(float(row["entry_price"] - row["stop_price"])) / float(
                row["entry_price"]
            )
            position_fraction = min(
                risk_fraction / stop_distance, max_position_fraction
            )
            open_notional[row_id] = equity * position_fraction
        else:
            notional = open_notional.pop(row_id, None)
            if notional is None:
                continue
            net_return = float(row["net_profit_usdt"]) / 30.0
            equity += notional * net_return
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, 100 * (peak - equity) / peak)
            equity_points.append({"date": timestamp, "equity": equity})
    return {
        "starting_equity": starting_equity,
        "final_equity": equity,
        "total_return_pct": 100 * (equity / starting_equity - 1),
        "max_drawdown_pct": max_drawdown,
        "equity_points": equity_points,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    trades = pd.read_csv(args.source, parse_dates=["entry_time", "exit_time"])
    trades["year"] = trades["entry_time"].dt.year
    development = trades[trades["year"].between(2021, 2023)]
    validation = trades[trades["year"] == 2024]
    final_test = trades[trades["year"] >= 2025]

    candidates = []
    grid = itertools.product(
        ("1h", "2h", "4h", "8h"),
        (-999.0, 0.0, 0.5),
        (-999.0, 0.0, 0.25),
        (999.0, 0.5, 1.0, 2.0),
        (999.0, 0.0, -0.25),
        (0.0, 0.8, 1.0),
    )
    for timeframe, ema50_min, slope20_min, approach_max, prior_body_max, volume_min in grid:
        candidate = {
            "timeframe": timeframe,
            "ema50_min": ema50_min,
            "slope20_min": slope20_min,
            "approach_max": approach_max,
            "prior_body_max": prior_body_max,
            "volume_min": volume_min,
        }
        scoped = apply_candidate(development, candidate)
        overall = stats(scoped)
        yearly = [stats(scoped[scoped["year"] == year]) for year in (2021, 2022, 2023)]
        pair_stats = [stats(scoped[scoped["pair"] == pair]) for pair in trades["pair"].unique()]
        if (
            overall["trades"] < 60
            or min(item["trades"] for item in yearly) < 12
            or min(item["trades"] for item in pair_stats) < 20
            or sum(item["net"] > 0 for item in yearly) < 2
            or overall["pf"] < 1.0
        ):
            continue
        candidate.update(
            {
                "dev_trades": overall["trades"],
                "dev_net": overall["net"],
                "dev_pf": overall["pf"],
                "dev_win_rate": overall["win_rate"],
                "dev_positive_years": sum(item["net"] > 0 for item in yearly),
                "dev_worst_year_net": min(item["net"] for item in yearly),
                "dev_min_pair_net": min(item["net"] for item in pair_stats),
            }
        )
        candidates.append(candidate)

    candidate_frame = pd.DataFrame(candidates)
    if candidate_frame.empty:
        raise RuntimeError("No development candidate passed the robustness gates")
    candidate_frame = candidate_frame.sort_values(
        ["dev_positive_years", "dev_pf", "dev_net"], ascending=False
    )

    validation_rows = []
    for candidate in candidate_frame.head(40).to_dict("records"):
        scoped = apply_candidate(validation, candidate)
        result = stats(scoped)
        candidate.update(
            {
                "validation_trades": result["trades"],
                "validation_net": result["net"],
                "validation_pf": result["pf"],
                "validation_win_rate": result["win_rate"],
            }
        )
        validation_rows.append(candidate)
    validated = pd.DataFrame(validation_rows)
    eligible = validated[
        (validated["validation_trades"] >= 15)
        & (validated["validation_net"] > 0)
        & (validated["validation_pf"] > 1.0)
    ].copy()
    if eligible.empty:
        raise RuntimeError("No top development candidate survived 2024 validation")
    eligible["robust_score"] = eligible[["dev_pf", "validation_pf"]].min(axis=1)
    locked = eligible.sort_values(
        ["robust_score", "validation_net", "dev_net"], ascending=False
    ).iloc[0].to_dict()

    period_rows = []
    for label, scoped_source in (
        ("development_2021_2023", development),
        ("validation_2024", validation),
        ("final_test_2025_2026", final_test),
        ("all_2021_2026", trades[trades["year"] >= 2021]),
    ):
        scoped = apply_candidate(scoped_source, locked)
        result = stats(scoped)
        portfolio = compounded_portfolio(scoped)
        period_rows.append(
            {
                "period": label,
                **result,
                "compounded_return_pct": portfolio["total_return_pct"],
                "final_equity": portfolio["final_equity"],
                "max_drawdown_pct": portfolio["max_drawdown_pct"],
            }
        )

    selected_all = apply_candidate(trades[trades["year"] >= 2021], locked)
    yearly_rows = []
    for year, scoped in selected_all.groupby("year"):
        result = stats(scoped)
        portfolio = compounded_portfolio(scoped)
        yearly_rows.append(
            {
                "year": year,
                **result,
                "compounded_return_pct": portfolio["total_return_pct"],
                "max_drawdown_pct": portfolio["max_drawdown_pct"],
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    candidate_frame.to_csv(args.output / "development_candidates.csv", index=False)
    eligible.to_csv(args.output / "validated_candidates.csv", index=False)
    pd.DataFrame([locked]).to_csv(args.output / "locked_parameters.csv", index=False)
    pd.DataFrame(period_rows).to_csv(args.output / "locked_period_results.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(args.output / "locked_yearly_results.csv", index=False)
    selected_all.to_csv(args.output / "locked_trades.csv", index=False)

    print("LOCKED PARAMETERS (chosen without 2025-2026 data)")
    print(pd.DataFrame([locked]).to_string(index=False))
    print("\nPERIOD RESULTS")
    print(pd.DataFrame(period_rows).to_string(index=False))
    print("\nYEARLY RESULTS")
    print(pd.DataFrame(yearly_rows).to_string(index=False))


if __name__ == "__main__":
    main()
