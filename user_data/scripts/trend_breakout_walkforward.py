"""Walk-forward research for an EMA200-regime Donchian breakout strategy."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from ema200_mtf_reaction_backtest import load_15m, resample_ohlcv
from ema200_exact_entry_backtest import TIMEFRAME_DELTA


DATA_DIR = Path("/freqtrade/user_data/data/binance/futures")
OUTPUT = Path("/freqtrade/user_data/backtest_results/trend_breakout_walkforward")
# The preliminary exact-entry study identified 4h as the only positive
# R:R=1.5 baseline.  Restrict the expensive walk-forward grid to that family.
TF_RULES = {"4h": ("4h", 16)}
FEE = 0.0005


def simulate_candidate(base, candles, pair, timeframe, candidate_id, channel, stop_atr, rr, horizon, slope_lookback):
    trades = []
    delta = TIMEFRAME_DELTA[timeframe]
    next_entry_time = candles.index[channel + slope_lookback + 1]
    detail_dates = base.index.values
    detail_high = base["high"].to_numpy(float)
    detail_low = base["low"].to_numpy(float)
    detail_close = base["close"].to_numpy(float)

    for i in range(max(channel, slope_lookback) + 1, len(candles)):
        candle_time = candles.index[i]
        if candle_time < next_entry_time or candle_time - candles.index[i - 1] != delta:
            continue
        ema = float(candles["ema200"].iloc[i - 1])
        ema_old = float(candles["ema200"].iloc[i - 1 - slope_lookback])
        close_previous = float(candles["close"].iloc[i - 1])
        atr = float(candles["atr14"].iloc[i - 1])
        if close_previous > ema and ema > ema_old:
            direction = "long"
            entry = float(candles["high"].iloc[i - channel : i].max())
            stop = entry - stop_atr * atr
            target = entry + rr * stop_atr * atr
            touched = float(candles["high"].iloc[i]) >= entry
        elif close_previous < ema and ema < ema_old:
            direction = "short"
            entry = float(candles["low"].iloc[i - channel : i].min())
            stop = entry + stop_atr * atr
            target = entry - rr * stop_atr * atr
            touched = float(candles["low"].iloc[i]) <= entry
        else:
            continue
        if not touched:
            continue

        end_time = candle_time + horizon * delta
        left = int(np.searchsorted(detail_dates, candle_time.to_datetime64(), side="left"))
        right = int(np.searchsorted(detail_dates, end_time.to_datetime64(), side="left"))
        if right <= left:
            continue
        filled = False
        entry_time = None
        exit_idx = right - 1
        exit_price = float(detail_close[exit_idx])
        reason = "time_exit"
        for j in range(left, right):
            if not filled:
                hit_entry = detail_high[j] >= entry if direction == "long" else detail_low[j] <= entry
                if not hit_entry:
                    continue
                filled = True
                entry_time = pd.Timestamp(detail_dates[j], tz="UTC")
            if direction == "long":
                hit_stop = detail_low[j] <= stop
                hit_target = detail_high[j] >= target
            else:
                hit_stop = detail_high[j] >= stop
                hit_target = detail_low[j] <= target
            if hit_stop:
                exit_idx, exit_price, reason = j, stop, "stop"
                break
            if hit_target:
                exit_idx, exit_price, reason = j, target, "target"
                break
        if not filled:
            continue
        exit_time = pd.Timestamp(detail_dates[exit_idx], tz="UTC")
        gross_return = exit_price / entry - 1 if direction == "long" else 1 - exit_price / entry
        net_return = gross_return - FEE * (1 + exit_price / entry)
        trades.append({
            "candidate_id": candidate_id, "pair": pair, "timeframe": timeframe,
            "channel": channel, "stop_atr": stop_atr, "rr": rr,
            "horizon": horizon, "slope_lookback": slope_lookback,
            "direction": direction, "entry_time": entry_time, "exit_time": exit_time,
            "entry_price": entry, "stop_price": stop, "target_price": target,
            "exit_price": exit_price, "exit_reason": reason, "net_return": net_return,
        })
        next_entry_time = exit_time + pd.Timedelta(minutes=15)
    return trades


def metrics(frame):
    wins = frame.loc[frame.net_return > 0, "net_return"].sum()
    losses = -frame.loc[frame.net_return < 0, "net_return"].sum()
    return {
        "trades": len(frame), "net": frame.net_return.sum(),
        "pf": wins / losses if losses else np.inf,
        "win_rate": 100 * frame.net_return.gt(0).mean() if len(frame) else 0,
    }


def portfolio(frame, initial=1000.0, risk=0.01, max_fraction=0.5):
    frame = frame.sort_values(["entry_time", "exit_time"]).copy()
    events = []
    for idx, row in frame.iterrows():
        events.extend([(row.entry_time, 1, idx), (row.exit_time, 0, idx)])
    events.sort(key=lambda value: (value[0], value[1]))
    equity = peak = initial
    max_dd = 0.0
    positions = {}
    points = []
    for timestamp, kind, idx in events:
        row = frame.loc[idx]
        if kind == 1:
            stop_distance = abs(row.entry_price - row.stop_price) / row.entry_price
            positions[idx] = equity * min(risk / stop_distance, max_fraction)
        else:
            notional = positions.pop(idx, None)
            if notional is None:
                continue
            equity += notional * row.net_return
            peak = max(peak, equity)
            max_dd = max(max_dd, 100 * (peak - equity) / peak)
            points.append({"date": timestamp, "equity": equity})
    return equity, 100 * (equity / initial - 1), max_dd, points


def main():
    market_data = []
    for source in sorted(DATA_DIR.glob("*-15m-futures.feather")):
        pair = source.name.split("-")[0].replace("_", "/", 1).replace("_", ":")
        base = load_15m(source)
        for timeframe, (rule, expected) in TF_RULES.items():
            market_data.append((pair, timeframe, base, resample_ohlcv(base, rule, expected)))

    candidate_specs = list(itertools.product(TF_RULES, (10, 20, 40, 60), (1.0, 1.5, 2.0), (1.5, 2.0), (20, 40), (10, 20)))
    all_trades = []
    for candidate_id, (timeframe, channel, stop_atr, rr, horizon, slope_lookback) in enumerate(candidate_specs):
        for pair, data_tf, base, candles in market_data:
            if data_tf == timeframe:
                all_trades.extend(simulate_candidate(base, candles, pair, timeframe, candidate_id, channel, stop_atr, rr, horizon, slope_lookback))
    trades = pd.DataFrame(all_trades)
    trades["year"] = trades.entry_time.dt.year

    development_rows = []
    for candidate_id, scoped_all in trades.groupby("candidate_id"):
        scoped = scoped_all[scoped_all.year.between(2021, 2023)]
        overall = metrics(scoped)
        years = [metrics(scoped[scoped.year == year]) for year in (2021, 2022, 2023)]
        pairs = [metrics(scoped[scoped.pair == pair]) for pair in trades.pair.unique()]
        if overall["trades"] < 30 or min(x["trades"] for x in years) < 5 or min(x["trades"] for x in pairs) < 8:
            continue
        first = scoped_all.iloc[0]
        development_rows.append({
            "candidate_id": candidate_id, "timeframe": first.timeframe,
            "channel": first.channel, "stop_atr": first.stop_atr, "rr": first.rr,
            "horizon": first.horizon, "slope_lookback": first.slope_lookback,
            "dev_trades": overall["trades"], "dev_net": overall["net"], "dev_pf": overall["pf"],
            "positive_dev_years": sum(x["net"] > 0 for x in years),
            "worst_dev_year": min(x["net"] for x in years),
            "positive_pairs": sum(x["net"] > 0 for x in pairs),
        })
    dev = pd.DataFrame(development_rows)
    robust = dev[(dev.positive_dev_years == 3) & (dev.positive_pairs == 2) & (dev.dev_pf > 1.05)].copy()
    if robust.empty:
        robust = dev[(dev.positive_dev_years >= 2) & (dev.positive_pairs == 2) & (dev.dev_pf > 1.1)].copy()
    robust = robust.sort_values(["positive_dev_years", "worst_dev_year", "dev_pf"], ascending=False).head(40)

    validation_rows = []
    for row in robust.itertuples():
        scoped = trades[(trades.candidate_id == row.candidate_id) & (trades.year == 2024)]
        result = metrics(scoped)
        validation_rows.append({**row._asdict(), "validation_trades": result["trades"], "validation_net": result["net"], "validation_pf": result["pf"]})
    validated = pd.DataFrame(validation_rows)
    eligible = validated[(validated.validation_trades >= 8) & (validated.validation_net > 0) & (validated.validation_pf > 1)].copy()
    if eligible.empty:
        raise RuntimeError("No breakout candidate survived validation")
    eligible["score"] = eligible[["dev_pf", "validation_pf"]].min(axis=1)
    locked = eligible.sort_values(["positive_dev_years", "score", "worst_dev_year"], ascending=False).iloc[0]

    chosen = trades[trades.candidate_id == locked.candidate_id].copy()
    results = []
    for label, years in (("development", (2021, 2023)), ("validation", (2024, 2024)), ("final_test", (2025, 2026)), ("all", (2021, 2026))):
        scoped = chosen[chosen.year.between(*years)]
        result = metrics(scoped)
        equity, ret, dd, _ = portfolio(scoped)
        results.append({"period": label, **result, "return_pct": ret, "final_equity": equity, "max_drawdown_pct": dd})
    yearly = []
    for year, scoped in chosen[chosen.year >= 2021].groupby("year"):
        result = metrics(scoped); _, ret, dd, _ = portfolio(scoped)
        yearly.append({"year": year, **result, "return_pct": ret, "max_drawdown_pct": dd})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    trades.to_csv(OUTPUT / "all_candidate_trades.csv", index=False)
    eligible.to_csv(OUTPUT / "validated_candidates.csv", index=False)
    pd.DataFrame([locked]).to_csv(OUTPUT / "locked_parameters.csv", index=False)
    pd.DataFrame(results).to_csv(OUTPUT / "period_results.csv", index=False)
    pd.DataFrame(yearly).to_csv(OUTPUT / "yearly_results.csv", index=False)
    chosen.to_csv(OUTPUT / "locked_trades.csv", index=False)
    print("LOCKED", locked.to_dict())
    print(pd.DataFrame(results).to_string(index=False))
    print(pd.DataFrame(yearly).to_string(index=False))


if __name__ == "__main__":
    main()
