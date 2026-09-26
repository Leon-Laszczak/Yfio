import pandas as pd
import pytest


@pytest.fixture
def make_ohlcv():
    """Factory for small deterministic OHLCV DataFrames used by backtest tests."""
    def _make(
        opens,
        highs=None,
        lows=None,
        closes=None,
        volumes=None,
        start="2026-01-01",
        freq="1D",
    ):
        opens = list(opens)
        n = len(opens)

        if highs is None:
            highs = [x + 1 for x in opens]
        if lows is None:
            lows = [x - 1 for x in opens]
        if closes is None:
            closes = list(opens)
        if volumes is None:
            volumes = [1_000] * n

        index = pd.date_range(start, periods=n, freq=freq)

        return pd.DataFrame(
            {
                "Open": opens,
                "High": list(highs),
                "Low": list(lows),
                "Close": list(closes),
                "Volume": list(volumes),
            },
            index=index,
        )

    return _make
