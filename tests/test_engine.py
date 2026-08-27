from pathlib import Path
import sys

path = Path(__file__).parent.parent
sys.path.append(str(path.resolve()))

import pandas as pd
import pytest

import app.backtest.backtest as backtest_module


def make_df():
    """
    Small deterministic OHLCV dataset.

    Important prices:
    - first BUY executed at second candle Open = 100
    - final SELL can be executed at fourth candle Open = 120
    """
    index = pd.date_range(
        "2026-01-01",
        periods=4,
        freq="1D",
    )

    return pd.DataFrame(
        {
            "Open": [90, 100, 110, 120],
            "High": [101, 112, 117, 122],
            "Low": [89, 99, 108, 118],
            "Close": [95, 110, 115, 120],
            "Volume": [1000, 1000, 1000, 1000],
        },
        index=index,
    )


# =========================================================
# NEGATIVE TRANSACTION COST
# =========================================================

def test_negative_transaction_cost_raises_error():
    with pytest.raises(
        ValueError,
        match="Transaction cost can't be nagative",
    ):
        backtest_module.backtest(
            ticker="TEST",
            period="1y",
            interval="1d",
            transaction_cost=-0.001,
        )


# =========================================================
# FULL BACKTEST: BUY -> HOLD -> SELL
# =========================================================

def test_full_backtest_buy_hold_sell(monkeypatch):
    df = make_df()

    monkeypatch.setattr(
        backtest_module,
        "fetch_data",
        lambda ticker, period, interval: df,
    )

    monkeypatch.setattr(
        backtest_module,
        "MIN_LENGTH",
        0,
    )

    signals = iter([
        "BUY",
        "HOLD",
        "SELL",
    ])

    monkeypatch.setattr(
        backtest_module,
        "strategy",
        lambda historical_df: next(signals),
    )

    metrics, history, trades = backtest_module.backtest(
        ticker="TEST",
        period="1y",
        interval="1d",
        transaction_cost=0,
    )

    # BUY signal at candle 0:
    # execution happens at candle 1 Open = 100
    #
    # 10 000 / 100 = 100 shares
    #
    # SELL signal at candle 2:
    # execution happens at candle 3 Open = 120
    #
    # final cash = 100 * 120 = 12 000

    assert history.iloc[0] == pytest.approx(10_000)
    assert history.iloc[1] == pytest.approx(11_000)
    assert history.iloc[2] == pytest.approx(11_500)
    assert history.iloc[3] == pytest.approx(12_000)

    assert len(trades) == 1

    trade = trades.iloc[0]

    assert trade["type"] == "LONG"
    assert trade["entry_price"] == pytest.approx(100)
    assert trade["close_price"] == pytest.approx(120)
    assert trade["amount"] == pytest.approx(100)
    assert trade["pnl"] == pytest.approx(2_000)

    assert metrics["Total PnL"] == pytest.approx(2_000)
    assert metrics["Percent PnL"] == pytest.approx(0.20)


# =========================================================
# FORCE CLOSE LONG AT END
# =========================================================

def test_open_long_is_force_closed_at_end(monkeypatch):
    df = make_df()

    monkeypatch.setattr(
        backtest_module,
        "fetch_data",
        lambda ticker, period, interval: df,
    )

    monkeypatch.setattr(
        backtest_module,
        "MIN_LENGTH",
        0,
    )

    # Strategy never explicitly closes the LONG.
    monkeypatch.setattr(
        backtest_module,
        "strategy",
        lambda historical_df: "BUY",
    )

    metrics, history, trades = backtest_module.backtest(
        ticker="TEST",
        period="1y",
        interval="1d",
        transaction_cost=0,
    )

    # LONG opens at second candle Open = 100.
    # It must be automatically closed at final Close = 120.

    assert len(trades) == 1

    trade = trades.iloc[0]

    assert trade["type"] == "LONG"
    assert trade["entry_price"] == pytest.approx(100)
    assert trade["close_price"] == pytest.approx(120)
    assert trade["pnl"] == pytest.approx(2_000)

    assert history.iloc[-1] == pytest.approx(12_000)
    assert metrics["Total PnL"] == pytest.approx(2_000)


# =========================================================
# FORCE CLOSE SHORT AT END
# =========================================================

def test_open_short_is_force_closed_at_end(monkeypatch):
    df = make_df()

    monkeypatch.setattr(
        backtest_module,
        "fetch_data",
        lambda ticker, period, interval: df,
    )

    monkeypatch.setattr(
        backtest_module,
        "MIN_LENGTH",
        0,
    )

    # Strategy never explicitly closes the SHORT.
    monkeypatch.setattr(
        backtest_module,
        "strategy",
        lambda historical_df: "SELL",
    )

    metrics, history, trades = backtest_module.backtest(
        ticker="TEST",
        period="1y",
        interval="1d",
        transaction_cost=0,
    )

    # SHORT opens:
    #
    # initial cash = 10 000
    # entry = 100
    # amount = 100
    #
    # cash after short sale = 20 000
    #
    # forced close at final Close = 120:
    # 20 000 - 100 * 120 = 8 000

    assert len(trades) == 1

    trade = trades.iloc[0]

    assert trade["type"] == "SHORT"
    assert trade["entry_price"] == pytest.approx(100)
    assert trade["close_price"] == pytest.approx(120)
    assert trade["pnl"] == pytest.approx(-2_000)

    assert history.iloc[-1] == pytest.approx(8_000)
    assert metrics["Total PnL"] == pytest.approx(-2_000)


# =========================================================
# MIN_LENGTH
# =========================================================

def test_min_length_is_respected(monkeypatch):
    index = pd.date_range(
        "2026-01-01",
        periods=6,
        freq="1D",
    )

    df = pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104, 105],
            "High": [101, 102, 103, 104, 105, 106],
            "Low": [99, 100, 101, 102, 103, 104],
            "Close": [100, 101, 102, 103, 104, 105],
            "Volume": [1000] * 6,
        },
        index=index,
    )

    monkeypatch.setattr(
        backtest_module,
        "fetch_data",
        lambda ticker, period, interval: df,
    )

    monkeypatch.setattr(
        backtest_module,
        "MIN_LENGTH",
        2,
    )

    received_lengths = []

    def fake_strategy(historical_df):
        received_lengths.append(
            len(historical_df)
        )
        return "HOLD"

    monkeypatch.setattr(
        backtest_module,
        "strategy",
        fake_strategy,
    )

    _, history, trades = backtest_module.backtest(
        ticker="TEST",
        period="1y",
        interval="1d",
        transaction_cost=0,
    )

    # range(2, len(df) - 1)
    #
    # i = 2, 3, 4
    #
    # strategy receives:
    # df[:3], df[:4], df[:5]

    assert received_lengths == [
        3,
        4,
        5,
    ]

    # history:
    # initial value + one value for every loop iteration
    #
    # len = len(df) - MIN_LENGTH

    assert len(history) == 4

    assert history.index.equals(
        df.index[2:]
    )

    assert (history == 10_000).all()

    assert trades.empty