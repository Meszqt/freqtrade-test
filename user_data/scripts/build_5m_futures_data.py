"""Build complete multi-timeframe OHLCV candles from local 1-minute data."""

from pathlib import Path

import pandas as pd


DATA_DIR = Path("/freqtrade/user_data/data/binance/futures")
TIMEFRAMES = {
    "5m": ("5min", 5),
    "15m": ("15min", 15),
    "1h": ("1h", 60),
    "2h": ("2h", 120),
    "4h": ("4h", 240),
    "8h": ("8h", 480),
    "12h": ("12h", 720),
    "1d": ("1D", 1440),
}


def main() -> None:
    for source in sorted(DATA_DIR.glob("*-1m-futures.feather")):
        frame = pd.read_feather(source)
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        frame = frame.sort_values("date").drop_duplicates("date").set_index("date")
        for timeframe, (resample_rule, expected_bars) in TIMEFRAMES.items():
            target = Path(
                str(source).replace(
                    "-1m-futures.feather", f"-{timeframe}-futures.feather"
                )
            )
            grouped = frame.resample(
                resample_rule, label="left", closed="left", origin="epoch"
            )
            result = grouped.agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
                source_bars=("close", "count"),
            )
            result = result[result["source_bars"] == expected_bars].drop(
                columns="source_bars"
            )
            result = result.reset_index()
            result.to_feather(target)
            print(
                f"{target.name}: {len(result)} candles, "
                f"{result.date.min()} -> {result.date.max()}"
            )


if __name__ == "__main__":
    main()
