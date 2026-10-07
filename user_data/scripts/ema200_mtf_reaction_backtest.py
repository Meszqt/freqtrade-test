"""Measure how often price rejects and reverses from EMA200 by timeframe.

This is an event study, not a PnL strategy backtest.  It deliberately uses no
indicator other than EMA200; ATR is only a volatility-normalized measuring
ruler for the outcome, never an entry filter.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TIMEFRAMES = {
    "1h": ("1h", 4),
    "2h": ("2h", 8),
    "4h": ("4h", 16),
    "8h": ("8h", 32),
    "12h": ("12h", 48),
    "1d": ("1D", 96),
}


def load_15m(path: Path) -> pd.DataFrame:
    dataframe = pd.read_feather(path)
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    dataframe = dataframe[list(required)].copy()
    dataframe["date"] = pd.to_datetime(dataframe["date"], utc=True)
    return (
        dataframe.sort_values("date")
        .drop_duplicates("date", keep="last")
        .set_index("date")
    )


def resample_ohlcv(dataframe: pd.DataFrame, rule: str, expected_bars: int) -> pd.DataFrame:
    resample_options = {"label": "left", "closed": "left"}
    if rule != "1D":
        resample_options["origin"] = "epoch"
    grouped = dataframe.resample(rule, **resample_options)
    result = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        source_bars=("close", "count"),
    )
    # Exclude partial/gappy candles so every timeframe is compared fairly.
    result = result[result["source_bars"] == expected_bars].copy()
    result["ema200"] = result["close"].ewm(
        span=200, adjust=False, min_periods=200
    ).mean()

    previous_close = result["close"].shift(1)
    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # ATR is used only to express reversal distance consistently across assets.
    result["atr14"] = true_range.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    return result.dropna(subset=["ema200", "atr14"])


def study_events(
    dataframe: pd.DataFrame,
    pair: str,
    timeframe: str,
    tolerance: float,
    target_atr: float,
    failure_atr: float,
    horizon: int,
) -> list[dict]:
    events: list[dict] = []
    if len(dataframe) <= horizon + 1:
        return events

    opens = dataframe["open"].to_numpy(dtype=float)
    highs = dataframe["high"].to_numpy(dtype=float)
    lows = dataframe["low"].to_numpy(dtype=float)
    closes = dataframe["close"].to_numpy(dtype=float)
    emas = dataframe["ema200"].to_numpy(dtype=float)
    atrs = dataframe["atr14"].to_numpy(dtype=float)
    dates = dataframe.index

    # A cooldown prevents the same cluster around EMA200 from being counted as
    # several independent observations while its outcome window is still open.
    next_allowed = 1
    for i in range(1, len(dataframe) - horizon):
        if i < next_allowed:
            continue

        upper = emas[i] * (1 + tolerance)
        lower = emas[i] * (1 - tolerance)
        intersects = lows[i] <= upper and highs[i] >= lower
        if not intersects:
            continue

        direction = None
        if closes[i - 1] > emas[i - 1] * (1 + tolerance):
            direction = "support"
        elif closes[i - 1] < emas[i - 1] * (1 - tolerance):
            direction = "resistance"
        if direction is None:
            continue

        if direction == "support":
            immediate = closes[i] >= emas[i]
            target = emas[i] + target_atr * atrs[i]
            failure = emas[i] - failure_atr * atrs[i]
            mfe = (highs[i + 1 : i + horizon + 1].max() - emas[i]) / atrs[i]
            mae = (emas[i] - lows[i + 1 : i + horizon + 1].min()) / atrs[i]
        else:
            immediate = closes[i] <= emas[i]
            target = emas[i] - target_atr * atrs[i]
            failure = emas[i] + failure_atr * atrs[i]
            mfe = (emas[i] - lows[i + 1 : i + horizon + 1].min()) / atrs[i]
            mae = (highs[i + 1 : i + horizon + 1].max() - emas[i]) / atrs[i]

        entry_price = opens[i + 1]
        if direction == "support":
            trade_eligible = failure < entry_price < target
        else:
            trade_eligible = target < entry_price < failure
        trade_eligible = bool(trade_eligible and immediate)

        simulation_exit_price = closes[i + horizon]
        simulation_exit_date = dates[i + horizon]
        simulation_exit_reason = "time_exit"
        if trade_eligible:
            for j in range(i + 1, i + horizon + 1):
                if direction == "support":
                    hit_target = highs[j] >= target
                    hit_failure = lows[j] <= failure
                else:
                    hit_target = lows[j] <= target
                    hit_failure = highs[j] >= failure
                if hit_failure:
                    simulation_exit_price = failure
                    simulation_exit_date = dates[j]
                    simulation_exit_reason = "stop"
                    break
                if hit_target:
                    simulation_exit_price = target
                    simulation_exit_date = dates[j]
                    simulation_exit_reason = "target"
                    break
        elif not immediate:
            simulation_exit_reason = "no_entry_no_rejection"
        else:
            simulation_exit_reason = "no_entry_gap_outside_bracket"

        confirmed = False
        outcome = "no_followthrough"
        if not immediate:
            outcome = "no_immediate_rejection"
        else:
            for j in range(i + 1, i + horizon + 1):
                if direction == "support":
                    hit_target = highs[j] >= target
                    hit_failure = lows[j] <= failure
                else:
                    hit_target = lows[j] <= target
                    hit_failure = highs[j] >= failure

                # Intrabar ordering is unknown.  If both levels occur in the
                # same candle, classify it conservatively as a failure.
                if hit_failure:
                    outcome = "invalidated"
                    break
                if hit_target:
                    confirmed = True
                    outcome = "confirmed_reversal"
                    break

        events.append(
            {
                "pair": pair,
                "timeframe": timeframe,
                "date": dates[i],
                "direction": direction,
                "tolerance_pct": tolerance * 100,
                "target_atr": target_atr,
                "failure_atr": failure_atr,
                "horizon_bars": horizon,
                "ema200": emas[i],
                "atr14": atrs[i],
                "open": opens[i],
                "high": highs[i],
                "low": lows[i],
                "close": closes[i],
                "immediate_rejection": bool(immediate),
                "confirmed_reversal": confirmed,
                "outcome": outcome,
                "mfe_atr": float(mfe),
                "mae_atr": float(mae),
                "entry_date": dates[i + 1],
                "entry_price": entry_price,
                "target_price": target,
                "failure_price": failure,
                "trade_eligible": trade_eligible,
                "simulation_exit_date": simulation_exit_date,
                "simulation_exit_price": simulation_exit_price,
                "simulation_exit_reason": simulation_exit_reason,
            }
        )
        next_allowed = i + horizon + 1

    return events


def wilson_lower_bound(successes: int, observations: int, z: float = 1.96) -> float:
    if observations == 0:
        return float("nan")
    probability = successes / observations
    denominator = 1 + z * z / observations
    centre = probability + z * z / (2 * observations)
    margin = z * math.sqrt(
        probability * (1 - probability) / observations
        + z * z / (4 * observations * observations)
    )
    return (centre - margin) / denominator


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    dimensions = ["tolerance_pct", "target_atr", "timeframe"]
    frames = []
    scopes = [
        (events, "ALL", "both"),
        *[(events[events["pair"] == pair], pair, "both") for pair in sorted(events["pair"].unique())],
        *[
            (events[events["direction"] == direction], "ALL", direction)
            for direction in ("support", "resistance")
        ],
    ]
    for scoped, pair_scope, direction_scope in scopes:
        grouped = scoped.groupby(dimensions, observed=True)
        summary = grouped.agg(
            events=("confirmed_reversal", "size"),
            immediate_rejections=("immediate_rejection", "sum"),
            confirmed_reversals=("confirmed_reversal", "sum"),
            median_mfe_atr=("mfe_atr", "median"),
            median_mae_atr=("mae_atr", "median"),
        ).reset_index()
        summary["pair_scope"] = pair_scope
        summary["direction_scope"] = direction_scope
        frames.append(summary)

    output = pd.concat(frames, ignore_index=True)
    output["immediate_rejection_rate_pct"] = (
        100 * output["immediate_rejections"] / output["events"]
    )
    output["confirmed_reversal_rate_pct"] = (
        100 * output["confirmed_reversals"] / output["events"]
    )
    output["wilson_95_lower_pct"] = output.apply(
        lambda row: 100
        * wilson_lower_bound(int(row["confirmed_reversals"]), int(row["events"])),
        axis=1,
    )
    return output[
        [
            "pair_scope",
            "direction_scope",
            *dimensions,
            "events",
            "immediate_rejections",
            "immediate_rejection_rate_pct",
            "confirmed_reversals",
            "confirmed_reversal_rate_pct",
            "wilson_95_lower_pct",
            "median_mfe_atr",
            "median_mae_atr",
        ]
    ]


def summarize_base_by_year(events: pd.DataFrame) -> pd.DataFrame:
    base = events[
        np.isclose(events["tolerance_pct"], 0.25)
        & np.isclose(events["target_atr"], 1.0)
    ].copy()
    base["year"] = pd.to_datetime(base["date"], utc=True).dt.year
    yearly = (
        base.groupby(["year", "timeframe"], observed=True)
        .agg(
            events=("confirmed_reversal", "size"),
            immediate_rejections=("immediate_rejection", "sum"),
            confirmed_reversals=("confirmed_reversal", "sum"),
        )
        .reset_index()
    )
    yearly["immediate_rejection_rate_pct"] = (
        100 * yearly["immediate_rejections"] / yearly["events"]
    )
    yearly["confirmed_reversal_rate_pct"] = (
        100 * yearly["confirmed_reversals"] / yearly["events"]
    )
    yearly["wilson_95_lower_pct"] = yearly.apply(
        lambda row: 100
        * wilson_lower_bound(int(row["confirmed_reversals"]), int(row["events"])),
        axis=1,
    )
    yearly["rank"] = yearly.groupby("year", observed=True)[
        "wilson_95_lower_pct"
    ].rank(method="min", ascending=False).astype(int)
    return yearly.sort_values(["year", "rank"])


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
        default=Path("/freqtrade/user_data/backtest_results/ema200_mtf"),
    )
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--failure-atr", type=float, default=0.5)
    args = parser.parse_args()

    source_files = sorted(args.data_dir.glob("*-15m-futures.feather"))
    if not source_files:
        raise FileNotFoundError(f"No 15m futures Feather files in {args.data_dir}")

    tolerances = (0.001, 0.0025, 0.005)
    targets = (0.5, 1.0, 1.5)
    all_events: list[dict] = []
    dataset_ranges: dict[str, dict] = {}

    for source in source_files:
        pair = source.name.split("-")[0].replace("_", "/", 1).replace("_", ":")
        base = load_15m(source)
        dataset_ranges[pair] = {
            "source": source.name,
            "source_rows": len(base),
            "first_15m": base.index.min().isoformat(),
            "last_15m": base.index.max().isoformat(),
            "timeframes": {},
        }
        for timeframe, (rule, expected_bars) in TIMEFRAMES.items():
            candles = resample_ohlcv(base, rule, expected_bars)
            dataset_ranges[pair]["timeframes"][timeframe] = {
                "usable_rows_after_warmup": len(candles),
                "first": candles.index.min().isoformat(),
                "last": candles.index.max().isoformat(),
            }
            for tolerance in tolerances:
                for target_atr in targets:
                    all_events.extend(
                        study_events(
                            candles,
                            pair,
                            timeframe,
                            tolerance,
                            target_atr,
                            args.failure_atr,
                            args.horizon,
                        )
                    )

    events = pd.DataFrame(all_events)
    if events.empty:
        raise RuntimeError("No EMA200 touch events were found")
    summary = summarize(events)

    base_ranking = summary[
        (summary["pair_scope"] == "ALL")
        & (summary["direction_scope"] == "both")
        & np.isclose(summary["tolerance_pct"], 0.25)
        & np.isclose(summary["target_atr"], 1.0)
    ].sort_values(
        ["wilson_95_lower_pct", "confirmed_reversal_rate_pct"], ascending=False
    )
    sensitivity = summary[
        (summary["pair_scope"] == "ALL")
        & (summary["direction_scope"] == "both")
    ].copy()
    sensitivity_ranks = sensitivity.groupby(
        ["tolerance_pct", "target_atr"], observed=True
    )["wilson_95_lower_pct"].rank(method="min", ascending=False)
    sensitivity["rank"] = sensitivity_ranks.astype(int)
    robustness = (
        sensitivity.groupby("timeframe", observed=True)
        .agg(
            average_rank=("rank", "mean"),
            first_place_count=("rank", lambda values: int((values == 1).sum())),
            mean_confirmed_rate_pct=("confirmed_reversal_rate_pct", "mean"),
            total_scenarios=("rank", "size"),
        )
        .reset_index()
        .sort_values(["average_rank", "mean_confirmed_rate_pct"], ascending=[True, False])
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output_dir / "events.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    base_ranking.to_csv(args.output_dir / "base_ranking.csv", index=False)
    robustness.to_csv(args.output_dir / "sensitivity_ranking.csv", index=False)
    yearly = summarize_base_by_year(events)
    yearly.to_csv(args.output_dir / "base_yearly.csv", index=False)
    metadata = {
        "study_type": "EMA200 touch/reversal event study",
        "ema_period": 200,
        "timeframes": list(TIMEFRAMES),
        "touch_tolerances_pct": [value * 100 for value in tolerances],
        "reversal_targets_atr": list(targets),
        "failure_threshold_atr": args.failure_atr,
        "outcome_horizon_bars": args.horizon,
        "base_case": {"touch_tolerance_pct": 0.25, "target_atr": 1.0},
        "ranking_metric": "95% Wilson lower bound of confirmed reversal rate",
        "datasets": dataset_ranges,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nBASE CASE: ±0.25% touch, 1 ATR reversal, 0.5 ATR failure, 5 bars")
    print(base_ranking.to_string(index=False))
    print("\nSENSITIVITY RANKING (9 tolerance/target scenarios)")
    print(robustness.to_string(index=False))
    print("\nBASE-CASE WINNER BY CALENDAR YEAR")
    print(yearly[yearly["rank"] == 1].to_string(index=False))
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
