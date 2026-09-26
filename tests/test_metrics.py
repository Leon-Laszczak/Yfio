from pathlib import Path
import sys

path = Path(__file__).parent.parent
sys.path.append(str(path.resolve()))

import numpy as np
import pandas as pd
import pytest

from app.backtest.metrics import MetricsComputer


def make_history(values, start="2026-01-01", freq="1D"):
    return pd.Series(
        values,
        index=pd.date_range(start, periods=len(values), freq=freq),
        dtype=float,
    )


def make_trades(pnls):
    return pd.DataFrame({"pnl": pnls})


def test_requires_at_least_two_history_points():
    history = make_history([10_000])
    trades = make_trades([])

    with pytest.raises(ValueError, match="Insufficient amount of data"):
        MetricsComputer(history, trades)


def test_history_requires_datetime_index():
    history = pd.Series([10_000.0, 10_100.0], index=[0, 1])
    trades = make_trades([])

    with pytest.raises((TypeError, AttributeError)):
        MetricsComputer(history, trades)


def test_compute_pnl():
    history = make_history([10_000, 9_000, 10_500, 10_900, 11_200, 11_000, 12_000])
    metrics = MetricsComputer(history, make_trades([]))

    pnl, pct_pnl = metrics.compute_pnl()

    assert pnl == pytest.approx(2_000)
    assert pct_pnl == pytest.approx(0.20)


def test_win_rate_uses_closed_trade_pnl():
    history = make_history([10_000, 10_100, 10_200])
    trades = make_trades([100, -50, 0, 25])
    metrics = MetricsComputer(history, trades)

    assert metrics.compute_win_rate() == pytest.approx(0.5)


def test_win_rate_is_zero_without_trades():
    history = make_history([10_000, 10_100, 10_200])
    metrics = MetricsComputer(history, make_trades([]))

    assert metrics.compute_win_rate() == 0.0


def test_profit_factor_uses_trade_pnl():
    history = make_history([10_000, 10_100, 10_200])
    trades = make_trades([100, 50, -30, -20])
    metrics = MetricsComputer(history, trades)

    assert metrics.compute_profit_factor() == pytest.approx(3.0)


def test_profit_factor_is_infinite_when_there_are_no_losses():
    history = make_history([10_000, 10_100, 10_200])
    trades = make_trades([100, 50])
    metrics = MetricsComputer(history, trades)

    assert metrics.compute_profit_factor() == float("inf")


def test_daily_returns_use_last_equity_value_of_each_day():
    history = pd.Series(
        [100.0, 110.0, 121.0, 133.1],
        index=pd.to_datetime(
            [
                "2026-01-01 10:00",
                "2026-01-01 15:00",
                "2026-01-02 10:00",
                "2026-01-03 10:00",
            ]
        ),
    )
    metrics = MetricsComputer(history, make_trades([]))

    # Daily last values: 110 -> 121 -> 133.1, so both daily returns are 10%.
    np.testing.assert_allclose(metrics.daily_ret, [0.10, 0.10])


def test_periods_per_year_matches_duration_formula():
    history = make_history([100, 101, 102, 103, 104], freq="30D")
    metrics = MetricsComputer(history, make_trades([]))

    duration = (history.index[-1] - history.index[0]).total_seconds()
    years = duration / (365.25 * 24 * 60 * 60)
    expected = (len(history) - 1) / years

    assert metrics.ppy == pytest.approx(expected)


def test_compute_metrics_contains_public_payload_keys():
    history = make_history(
        [10_000, 10_100, 10_050, 10_200, 10_150, 10_300, 10_250, 10_400]
    )
    trades = make_trades([100, -50, 200])
    metrics = MetricsComputer(history, trades)

    payload = metrics.compute_metrics()

    expected_keys = {
        "Total PnL",
        "Percent PnL",
        "CAGR",
        "Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Calmar Ratio",
        "Max Drawdown",
        "Max Recovery Time",
        "Mean Recovery Time",
        "VaR 1d 95%",
        "VaR 1d 99%",
        "CVaR 1d 95%",
        "CVaR 1d 99%",
        "Win Rate",
        "Profit Factor",
        "Skewness",
        "Kurtosis",
    }

    assert set(payload) == expected_keys
    assert payload["Total PnL"] == pytest.approx(400)
    assert payload["Percent PnL"] == pytest.approx(0.04)
    assert payload["Win Rate"] == pytest.approx(2 / 3)

def test_empty_trades():
    trades = make_trades([])
    history = make_history([10_000, 10_000, 10_000, 10_000, 10_000, 10_000, 10_000, 10_000,])

    metrics = MetricsComputer(history,trades).compute_metrics()

    expected_keys = {
            "Total PnL",
            "Percent PnL",
            "CAGR",
            "Volatility",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Calmar Ratio",
            "Max Drawdown",
            "Max Recovery Time",
            "Mean Recovery Time",
            "VaR 1d 95%",
            "VaR 1d 99%",
            "CVaR 1d 95%",
            "CVaR 1d 99%",
            "Win Rate",
            "Profit Factor",
            "Skewness",
            "Kurtosis",
        }

    assert set(metrics) == expected_keys

    for key in expected_keys:
        assert metrics[key] == 0 or np.isnan(metrics[key])
    