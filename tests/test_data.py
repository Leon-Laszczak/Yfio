import pandas as pd
import pytest

from pathlib import Path
import sys

path = Path(__file__).parent.parent
sys.path.append(str(path.resolve()))

from app.backtest.data import fetch_data


def make_ohlcv():
    index = pd.date_range(
        "2026-01-01",
        periods=3,
        freq="D",
    )

    return pd.DataFrame({
        "Open": [100, 101, 102],
        "High": [102, 103, 104],
        "Low": [99, 100, 101],
        "Close": [101, 102, 103],
        "Volume": [1000, 1100, 1200],
    }, index=index)


def test_fetch_data_success(monkeypatch):
    df = make_ohlcv()

    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def history(self, period, interval):
            return df

    monkeypatch.setattr(
        "app.backtest.data.yf.Ticker",
        FakeTicker,
    )

    result = fetch_data(
        "AAPL",
        period="1y",
        interval="1d",
    )

    assert len(result) == 3

    assert list(result.columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]


def test_fetch_data_sorts_index(monkeypatch):
    df = make_ohlcv().sort_index(ascending=False)

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, period, interval):
            return df

    monkeypatch.setattr(
        "app.backtest.data.yf.Ticker",
        FakeTicker,
    )

    result = fetch_data("AAPL")

    assert result.index.is_monotonic_increasing


def test_fetch_data_removes_duplicate_dates(monkeypatch):
    df = make_ohlcv()

    duplicated = pd.concat([
        df,
        df.iloc[[1]],
    ])

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, period, interval):
            return duplicated

    monkeypatch.setattr(
        "app.backtest.data.yf.Ticker",
        FakeTicker,
    )

    result = fetch_data("AAPL")

    assert not result.index.duplicated().any()


def test_fetch_data_raises_for_missing_ticker(monkeypatch):
    empty = pd.DataFrame()

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, period, interval):
            return empty

    monkeypatch.setattr(
        "app.backtest.data.yf.Ticker",
        FakeTicker,
    )

    with pytest.raises(ValueError):
        fetch_data("DOES_NOT_EXIST")