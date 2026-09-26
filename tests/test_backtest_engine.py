from pathlib import Path
import sys

path = Path(__file__).parent.parent
sys.path.append(str(path.resolve()))

import pandas as pd
import numpy as np
import random
import math
import pytest

import app.backtest.backtest as bt

def generate_random_market(n=1000, seed=42):
    rng = np.random.default_rng(seed)

    returns = rng.normal(
        loc=0,
        scale=0.01,
        size=n,
    )

    close = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "Open": close,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
    })

    return df

class DummyMetricsComputer:
    """Keeps engine tests isolated from MetricsComputer implementation details."""

    def __init__(self, history, trades):
        self.history = history
        self.trades = trades

    def compute_metrics(self):
        return {
            "final_equity": float(self.history.iloc[-1]),
            "trade_count": len(self.trades),
        }


@pytest.fixture(autouse=True)
def isolate_metrics(monkeypatch):
    monkeypatch.setattr(bt, "MetricsComputer", DummyMetricsComputer)
    monkeypatch.setattr(bt, "MIN_LENGTH", 0)


def patch_market(monkeypatch, df, signals):
    monkeypatch.setattr(bt, "fetch_data", lambda ticker, period, interval: df)

    def fake_strategy(historical_df):
        # With MIN_LENGTH=0, historical lengths are 1, 2, ..., n-1.
        return signals[len(historical_df) - 1]

    monkeypatch.setattr(bt, "strategy", fake_strategy)


def test_multiple_long_trades_can_be_open_simultaneously(monkeypatch, make_ohlcv):
    df = make_ohlcv(
        opens=[100, 100, 100, 100],
        highs=[101, 101, 101, 101],
        lows=[99, 99, 99, 99],
        closes=[100, 100, 100, 100],
    )
    patch_market(monkeypatch, df, ["BUY", "BUY", "HOLD"])

    metrics, history, trades = bt.backtest(
        "TEST",
        "1y",
        "1d",
        transaction_cost=0,
        position_size=0.10,
        
        take_profit=0.50,
        stop_loss=0.50,
    )

    assert len(trades) == 2
    assert trades["type"].tolist() == ["LONG", "LONG"]
    assert set(trades["exit_reason"]) == {"FORCED_CLOSE"}
    assert trades["entry_date"].tolist() == [df.index[1], df.index[2]]
    assert trades["close_date"].tolist() == [df.index[-1], df.index[-1]]
    assert history.iloc[-1] == pytest.approx(10_000)
    assert metrics["trade_count"] == 2


def test_buy_and_sell_open_independent_positions(monkeypatch, make_ohlcv):
    df = make_ohlcv(
        opens=[100, 100, 100, 100],
        highs=[101, 101, 101, 101],
        lows=[99, 99, 99, 99],
        closes=[100, 100, 100, 100],
    )
    patch_market(monkeypatch, df, ["BUY", "SELL", "HOLD"])

    _, _, trades = bt.backtest(
        "TEST",
        "1y",
        "1d",
        transaction_cost=0,
        position_size=0.10,
        take_profit=0.50,
        stop_loss=0.50,
    )

    assert trades["type"].tolist() == ["LONG", "SHORT"]
    assert trades["exit_reason"].tolist() == ["FORCED_CLOSE", "FORCED_CLOSE"]


def test_strategy_signal_does_not_close_existing_trade(monkeypatch, make_ohlcv):
    df = make_ohlcv(
        opens=[100, 100, 100, 100],
        highs=[101, 101, 101, 101],
        lows=[99, 99, 99, 99],
        closes=[100, 100, 100, 100],
    )
    patch_market(monkeypatch, df, ["BUY", "SELL", "HOLD"])

    _, _, trades = bt.backtest(
        "TEST",
        "1y",
        "1d",
        transaction_cost=0,
        position_size=0.10,
        take_profit=0.50,
        stop_loss=0.50,
    )

    first = trades.iloc[0]
    assert first["type"] == "LONG"
    assert first["close_date"] == df.index[-1]
    assert first["exit_reason"] == "FORCED_CLOSE"


def test_one_trade_can_tp_while_another_stays_open(monkeypatch, make_ohlcv):
    df = make_ohlcv(
        opens=[100, 100, 120, 120],
        highs=[101, 105, 125, 121],
        lows=[99, 99, 118, 119],
        closes=[100, 102, 121, 120],
    )
    patch_market(monkeypatch, df, ["BUY", "BUY", "HOLD"])

    _, _, trades = bt.backtest(
        "TEST",
        "1y",
        "1d",
        transaction_cost=0,
        position_size=0.10,
        take_profit=0.10,
        stop_loss=0.20,
    )

    assert len(trades) == 2

    first = trades.iloc[0]
    second = trades.iloc[1]

    # First trade entered at 100. The next candle opens at 120, through its TP 110.
    assert first["entry_price"] == pytest.approx(100)
    assert first["close_price"] == pytest.approx(120)
    assert first["exit_reason"] == "TP"

    # Second trade enters at 120 on that same candle and survives to the end.
    assert second["entry_price"] == pytest.approx(120)
    assert second["exit_reason"] == "FORCED_CLOSE"


def test_new_trade_can_hit_stop_on_entry_candle(monkeypatch, make_ohlcv):
    df = make_ohlcv(
        opens=[100, 100, 100],
        highs=[101, 102, 101],
        lows=[99, 94, 99],
        closes=[100, 96, 100],
    )
    patch_market(monkeypatch, df, ["BUY", "HOLD"])

    _, _, trades = bt.backtest(
        "TEST",
        "1y",
        "1d",
        transaction_cost=0,
        position_size=0.10,

        take_profit=0.10,
        stop_loss=0.05,
    )

    assert len(trades) == 1
    assert trades.iloc[0]["close_date"] == df.index[1]
    assert trades.iloc[0]["close_price"] == pytest.approx(95)
    assert trades.iloc[0]["exit_reason"] == "SL"


def test_final_history_includes_forced_close_exit_fee(monkeypatch, make_ohlcv):
    df = make_ohlcv(
        opens=[100, 100, 100],
        highs=[101, 101, 101],
        lows=[99, 99, 99],
        closes=[100, 100, 100],
    )
    patch_market(monkeypatch, df, ["BUY", "HOLD"])

    _, history, trades = bt.backtest(
        "TEST",
        "1y",
        "1d",
        transaction_cost=0.001,
        position_size=0.10,
        take_profit=0.50,
        stop_loss=0.50,
    )

    assert len(trades) == 1

    # 10% of 10,000 (= 1,000) is consumed at entry including the entry fee.
    # With a flat price, final loss is entry fee + exit fee.
    amount = trades.iloc[0]["amount"]
    expected_final = 10_000 - (amount * 100 * 0.001) - (amount * 100 * 0.001)

    assert history.iloc[-1] == pytest.approx(expected_final)
    assert trades.iloc[0]["pnl"] == pytest.approx(expected_final - 10_000)


def test_history_index_matches_history_length(monkeypatch, make_ohlcv):
    df = make_ohlcv(opens=[100, 101, 102, 103, 104])
    patch_market(monkeypatch, df, ["HOLD"] * 4)

    _, history, trades = bt.backtest("TEST", "1y", "1d")

    assert len(history) == len(df)
    assert history.index.equals(df.index)
    assert trades.empty


@pytest.mark.parametrize(
    "position_size",
    [
        -1,
        -0.000001,
        0,
        1.000001,
        2,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_position_size(position_size):
    with pytest.raises(ValueError):
        bt.backtest("TEST", "1y", "1d", position_size=position_size)

@pytest.mark.parametrize(
    "transaction_cost",
    [
        -1,
        -0.000001,
        1.000001,
        2,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_transaction_cost(transaction_cost):
    with pytest.raises(ValueError):
        bt.backtest("TEST", "1y", "1d", transaction_cost=transaction_cost)

@pytest.mark.parametrize(
    "stop_loss",
    [
        -1,
        -0.000001,
        0,
        1.000001,
        2,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_stop_loss(stop_loss):
    with pytest.raises(ValueError):
        bt.backtest("TEST", "1y", "1d", stop_loss=stop_loss)

def test_insufficient_data_raises(monkeypatch, make_ohlcv):
    df = make_ohlcv(opens=[100, 100, 100])
    monkeypatch.setattr(bt, "fetch_data", lambda ticker, period, interval: df)
    monkeypatch.setattr(bt, "MIN_LENGTH", 3)

    with pytest.raises(ValueError, match="Not enough data"):
        bt.backtest("TEST", "1y", "1d")


def test_unknown_strategy_signal_raises(monkeypatch, make_ohlcv):
    df = make_ohlcv(opens=[100, 100, 100])
    monkeypatch.setattr(bt, "fetch_data", lambda ticker, period, interval: df)
    monkeypatch.setattr(bt, "strategy", lambda historical_df: "WAIT")

    with pytest.raises(ValueError, match="Unknown signal was passed to the backtesting engine"):
        bt.backtest("TEST", "1y", "1d")

def test_trade_executed_on_next_candle_open(monkeypatch,make_ohlcv):
    df = make_ohlcv(
        opens=[100, 101, 103],
        highs=[101, 105, 105],
        lows=[99, 99, 99],
        closes=[100, 102, 104],
    )

    patch_market(monkeypatch,df,["BUY","HOLD","HOLD"])

    _, _, trades = bt.backtest("TEST","1y", "1d")

    assert len(trades) == 1
    assert trades["entry_price"].iloc[0] == 101

def test_take_profit_triggers_exactly_boundry(monkeypatch,make_ohlcv):
    df = make_ohlcv(
        opens=[100, 100],
        highs=[100, 110],
        lows=[100, 100],
        closes=[100, 105],
    )

    patch_market(monkeypatch,df,["BUY","HOLD"])

    _, _, trades = bt.backtest("TEST", "1y", "1d",position_size = 1,take_profit=0.1)

    assert len(trades) == 1
    assert trades["close_price"].iloc[0] == pytest.approx(110)

def test_final_equity_matches_realized_pnl(monkeypatch):
    random.seed(42)
    df = generate_random_market()
    patch_market(monkeypatch,df,[random.choice(["BUY","HOLD","SELL"]) for _ in range(1000)])
    metrics, history, trades = bt.backtest("TEST", "1y", "1d", position_size=1,transaction_cost=0)

    initial_cash = 10_000

    total_pnl = trades["pnl"].sum()

    expected_equity = initial_cash + total_pnl
    assert history.iloc[-1] == pytest.approx(
        expected_equity,
        rel=1e-9,
        abs=1e-6,
        )    

def test_backtest_is_deterministic(monkeypatch,make_ohlcv):
    random.seed(42)
    df = make_ohlcv(
            opens=[100 * (i/100) for i in range(100)],
            highs=[101 * (i/100) for i in range(100)],
            lows=[99 * (i/100) for i in range(100)],
            closes=[100 * (i/100) for i in range(100)],
        )

    patch_market(monkeypatch,df,[random.choice(["BUY","HOLD","SELL"]) for _ in range(100)])

    result_1 = bt.backtest("TEST", "1y", "1d")
    result_2 = bt.backtest("TEST", "1y", "1d")

    assert result_1[0] == result_2[0]
    pd.testing.assert_series_equal(
        result_1[1],
        result_2[1],
    )
    pd.testing.assert_frame_equal(
        result_1[2],
        result_2[2],
    )

def test_equity_curve_regression(monkeypatch,make_ohlcv):
    df = make_ohlcv(
            opens=[100, 100, 99, 120, 120],
            highs=[101, 105, 100, 125, 121],
            lows=[99, 99, 87, 118, 119],
            closes=[100, 98, 102, 121, 120],
        )
    patch_market(monkeypatch,df,["BUY","HOLD","HOLD","HOLD","HOLD"])

    _, history, _ = bt.backtest("TEST", "1y", "1d", transaction_cost=0, position_size=1, take_profit=1e9, stop_loss=1)

    expected = [
        10_000,
        9_800,
        10_200,
        12_100,
        12_000,
    ]

    assert history.tolist() == pytest.approx(expected)

@pytest.mark.parametrize(
    "seed",
    range(20),
)
def test_large_dataset(monkeypatch,seed):
    random.seed(42)
    df = generate_random_market(
        n = 10_000,
        seed = seed,
    )
    patch_market(monkeypatch,df,[random.choice(["BUY","HOLD","SELL"]) for _ in range(10_000)])

    _, history, _ = bt.backtest("TEST", "1y", "1d")

    assert math.isfinite(history.iloc[-1])
    assert history.iloc[-1] >= 0
    assert len(history) == len(df)
