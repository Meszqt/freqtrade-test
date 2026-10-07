"""Report portfolio risk scenarios for the locked EMA200 pullback trades."""

from pathlib import Path

import pandas as pd

from ema200_strategy_optimizer import compounded_portfolio


SOURCE = Path(
    "/freqtrade/user_data/backtest_results/ema200_research_rr15_stop125/"
    "optimized/locked_trades.csv"
)
OUTPUT = SOURCE.parent / "risk_scenarios.csv"


def main() -> None:
    trades = pd.read_csv(SOURCE, parse_dates=["entry_time", "exit_time"])
    trades["net_return"] = trades["net_profit_usdt"] / 30.0
    trades["year"] = trades["entry_time"].dt.year
    rows = []
    for name, risk, cap in (
        ("conservative", 0.01, 0.50),
        ("moderate", 0.02, 1.00),
        ("aggressive", 0.03, 1.50),
    ):
        for period, scoped in (
            ("all_2021_2026", trades),
            ("final_test_2025_2026", trades[trades["year"] >= 2025]),
        ):
            result = compounded_portfolio(
                scoped,
                starting_equity=1000.0,
                risk_fraction=risk,
                max_position_fraction=cap,
            )
            rows.append(
                {
                    "scenario": name,
                    "risk_per_trade_pct": risk * 100,
                    "max_position_fraction_pct": cap * 100,
                    "period": period,
                    "trades": len(scoped),
                    "return_pct": result["total_return_pct"],
                    "final_equity": result["final_equity"],
                    "max_drawdown_pct": result["max_drawdown_pct"],
                }
            )
    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(OUTPUT, index=False)
    print(result_frame.to_string(index=False))

    breakdown = []
    for dimension, grouped in (
        ("year", trades.groupby("year")),
        ("pair", trades.groupby("pair")),
        ("direction", trades.groupby("direction")),
        ("year_direction", trades.groupby(["year", "direction"])),
    ):
        for value, scoped in grouped:
            result = compounded_portfolio(
                scoped,
                starting_equity=1000.0,
                risk_fraction=0.02,
                max_position_fraction=1.0,
            )
            breakdown.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "trades": len(scoped),
                    "return_pct": result["total_return_pct"],
                    "max_drawdown_pct": result["max_drawdown_pct"],
                }
            )
    breakdown_frame = pd.DataFrame(breakdown)
    breakdown_frame.to_csv(SOURCE.parent / "moderate_breakdown.csv", index=False)
    print("\nMODERATE BREAKDOWN")
    print(breakdown_frame.to_string(index=False))

    volatility_rows = []
    for maximum_atr_pct in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 999.0):
        filtered = trades[trades["atr_pct"] <= maximum_atr_pct]
        for period, scoped in (
            ("development_2021_2023", filtered[filtered["year"].between(2021, 2023)]),
            ("validation_2024", filtered[filtered["year"] == 2024]),
            ("final_test_2025_2026", filtered[filtered["year"] >= 2025]),
        ):
            result = compounded_portfolio(
                scoped, risk_fraction=0.02, max_position_fraction=1.0
            )
            volatility_rows.append(
                {
                    "maximum_atr_pct": maximum_atr_pct,
                    "period": period,
                    "trades": len(scoped),
                    "return_pct": result["total_return_pct"],
                    "max_drawdown_pct": result["max_drawdown_pct"],
                }
            )
    volatility_frame = pd.DataFrame(volatility_rows)
    volatility_frame.to_csv(SOURCE.parent / "volatility_filter_sensitivity.csv", index=False)
    print("\nVOLATILITY FILTER SENSITIVITY")
    print(volatility_frame.to_string(index=False))

    locked_low_volatility = trades[trades["atr_pct"] <= 1.75]
    low_volatility_rows = []
    for label, scoped in [
        (str(year), group) for year, group in locked_low_volatility.groupby("year")
    ] + [
        (f"pair:{pair}", group)
        for pair, group in locked_low_volatility.groupby("pair")
    ] + [("all", locked_low_volatility)]:
        result = compounded_portfolio(
            scoped, risk_fraction=0.02, max_position_fraction=1.0
        )
        low_volatility_rows.append(
            {
                "period": label,
                "trades": len(scoped),
                "return_pct": result["total_return_pct"],
                "final_equity": result["final_equity"],
                "max_drawdown_pct": result["max_drawdown_pct"],
            }
        )
    low_volatility_frame = pd.DataFrame(low_volatility_rows)
    low_volatility_frame.to_csv(SOURCE.parent / "locked_low_volatility_results.csv", index=False)
    print("\nLOCKED ATR <= 1.75% RESULTS")
    print(low_volatility_frame.to_string(index=False))


if __name__ == "__main__":
    main()
