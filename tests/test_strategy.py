from pathlib import Path
import sys

path = Path(__file__).parent.parent
sys.path.append(str(path.resolve()))

import pandas as pd

from app.strategy.strategy import MIN_LENGTH, strategy


def make_df():
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1_000, 1_100],
        },
        index=pd.date_range("2026-01-01", periods=2, freq="1D"),
    )


def test_min_length_is_non_negative_integer():
    assert isinstance(MIN_LENGTH, int)
    assert MIN_LENGTH >= 0


def test_placeholder_strategy_returns_supported_signal():
    df = make_df()

    for _ in range(50):
        assert strategy(df) in {"BUY", "HOLD", "SELL"}


def test_strategy_does_not_mutate_input_dataframe():
    df = make_df()
    before = df.copy(deep=True)

    strategy(df)

    pd.testing.assert_frame_equal(df, before)
