from pathlib import Path
import sys

path = Path(__file__).parent.parent
sys.path.append(str(path.resolve()))

import pandas as pd
import pytest

import app.backtest.data as data_module


def sample_frame(index=None):
    if index is None:
        index = pd.to_datetime(["2026-01-02", "2026-01-01"])

    return pd.DataFrame(
        {
            "Open": [101.0, 100.0],
            "High": [102.0, 101.0],
            "Low": [100.0, 99.0],
            "Close": [101.5, 100.5],
            "Volume": [1_000, 900],
            "Ignored": [1, 2],
        },
        index=index,
    )


def test_fetch_data_returns_only_required_columns_and_sorts_index(monkeypatch):
    frame = sample_frame()

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period, interval):
            return frame.copy()

    monkeypatch.setattr(data_module.yf, "Ticker", FakeTicker)

    result = data_module.fetch_data("TEST", "1y", "1d")

    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert result.index.is_monotonic_increasing


def test_fetch_data_removes_duplicate_timestamps_keep_last(monkeypatch):
    index = pd.to_datetime(
        ["2026-01-01", "2026-01-01", "2026-01-02"]
    )
    frame = pd.DataFrame(
        {
            "Open": [100, 111, 120],
            "High": [101, 112, 121],
            "Low": [99, 110, 119],
            "Close": [100, 111, 120],
            "Volume": [1, 2, 3],
        },
        index=index,
    )

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period, interval):
            return frame.copy()

    monkeypatch.setattr(data_module.yf, "Ticker", FakeTicker)

    result = data_module.fetch_data("TEST", "1y", "1d")

    assert len(result) == 2
    assert result.loc[pd.Timestamp("2026-01-01"), "Open"] == 111

def test_fetch_data_tries_suffixes(monkeypatch):
    calls = []
    valid = sample_frame()

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period, interval):
            calls.append(self.symbol)
            if self.symbol == "ABC.L":
                return valid.copy()
            return pd.DataFrame()

    monkeypatch.setattr(data_module.yf, "Ticker", FakeTicker)

    result = data_module.fetch_data("ABC", "1y", "1d")

    assert not result.empty
    assert "ABC" in calls
    assert "ABC.L" in calls


def test_fetch_data_skips_provider_exceptions(monkeypatch):
    valid = sample_frame()

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period, interval):
            if self.symbol == "ABC":
                raise RuntimeError("provider failure")
            if self.symbol == "ABC.L":
                return valid.copy()
            return pd.DataFrame()

    monkeypatch.setattr(data_module.yf, "Ticker", FakeTicker)

    result = data_module.fetch_data("ABC", "1y", "1d")

    assert not result.empty


def test_fetch_data_raises_if_every_candidate_is_empty(monkeypatch):
    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period, interval):
            return pd.DataFrame()

    monkeypatch.setattr(data_module.yf, "Ticker", FakeTicker)

    with pytest.raises(
        ValueError,
        match="No valid data found",
    ):
        data_module.fetch_data("BAD", "1y", "1d")
