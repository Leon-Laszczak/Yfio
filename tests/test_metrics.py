from pathlib import Path
import sys

path = Path(__file__).parent.parent
sys.path.append(str(path.resolve()))

import numpy as np
import pandas as pd
import pytest

from app.backtest.metrics import MetricsComputer


def make_history(values, freq="1D"):
    index = pd.date_range(
        "2025-01-01",
        periods=len(values),
        freq=freq,
    )

    return pd.Series(values, index=index)


def make_trades(pnls):
    return pd.DataFrame({
        "pnl": pnls
    })


# =========================================================
# INITIALIZATION
# =========================================================

def test_requires_at_least_two_history_records():
    history = make_history([100])
    trades = make_trades([])

    with pytest.raises(ValueError):
        MetricsComputer(history, trades)


def test_history_requires_datetime_index():
    history = pd.Series([100, 110])
    trades = make_trades([])

    with pytest.raises(TypeError):
        MetricsComputer(history, trades)


# =========================================================
# PERIODS PER YEAR
# =========================================================

def test_periods_per_year_daily_data():
    history = make_history(
        [100] * 366,
        freq="1D",
    )

    metrics = MetricsComputer(
        history,
        make_trades([]),
    )

    assert metrics.ppy == pytest.approx(
        365.25,
        rel=0.01,
    )


def test_periods_per_year_hourly_data():
    history = make_history(
        [100] * 25,
        freq="1h",
    )

    metrics = MetricsComputer(
        history,
        make_trades([]),
    )

    expected = 24 * 365.25

    assert metrics.ppy == pytest.approx(expected)


# =========================================================
# RETURNS
# =========================================================

def test_periodic_returns():
    history = make_history([
        100,
        110,
        121,
    ])

    metrics = MetricsComputer(
        history,
        make_trades([]),
    )

    expected = np.array([
        0.10,
        0.10,
    ])

    np.testing.assert_allclose(
        metrics.ret,
        expected,
    )


# =========================================================
# DAILY RETURNS
# =========================================================

def test_daily_returns_uses_last_value_of_day():
    index = pd.to_datetime([
        "2025-01-01 10:00",
        "2025-01-01 16:00",
        "2025-01-02 10:00",
        "2025-01-02 16:00",
        "2025-01-03 16:00",
    ])

    history = pd.Series(
        [
            100,
            110,
            120,
            121,
            133.1,
        ],
        index=index,
    )

    metrics = MetricsComputer(
        history,
        make_trades([]),
    )

    # Daily closes:
    # day 1 -> 110
    # day 2 -> 121
    # day 3 -> 133.1
    #
    # returns:
    # 121 / 110 - 1 = 10%
    # 133.1 / 121 - 1 = 10%

    expected = np.array([
        0.10,
        0.10,
    ])

    np.testing.assert_allclose(
        metrics.daily_ret,
        expected,
    )


def test_daily_returns_skip_missing_days():
    index = pd.to_datetime([
        "2025-01-01",
        "2025-01-03",
    ])

    history = pd.Series(
        [100, 110],
        index=index,
    )

    metrics = MetricsComputer(
        history,
        make_trades([]),
    )

    assert len(metrics.daily_ret) == 1
    assert metrics.daily_ret[0] == pytest.approx(0.10)


# =========================================================
# PNL
# =========================================================

def test_compute_pnl_profit():
    history = make_history([
        1,
        0.9,
        1.05,
        1.09,
        1.12,
        1.1,
        1.2,
    ])

    metrics = MetricsComputer(
        history,
        make_trades([]),
    )

    pnl, pct_pnl = metrics.compute_pnl()

    assert pnl == pytest.approx(0.2)
    assert pct_pnl == pytest.approx(0.2)


def test_compute_pnl_loss():
    history = make_history([
        100,
        90,
    ])

    metrics = MetricsComputer(
        history,
        make_trades([]),
    )

    pnl, pct_pnl = metrics.compute_pnl()

    assert pnl == -10
    assert pct_pnl == pytest.approx(-0.10)


# =========================================================
# WIN RATE
# =========================================================

def test_win_rate():
    metrics = MetricsComputer(
        make_history([100, 101]),
        make_trades([
            100,
            -50,
            200,
            -20,
        ]),
    )

    assert metrics.compute_win_rate() == 0.5


def test_win_rate_empty_trades():
    metrics = MetricsComputer(
        make_history([100, 101]),
        make_trades([]),
    )

    assert metrics.compute_win_rate() == 0


def test_break_even_trade_is_not_win():
    metrics = MetricsComputer(
        make_history([100, 101]),
        make_trades([
            100,
            0,
            -50,
        ]),
    )

    assert metrics.compute_win_rate() == pytest.approx(1 / 3)


# =========================================================
# PROFIT FACTOR
# =========================================================

def test_profit_factor():
    metrics = MetricsComputer(
        make_history([100, 101]),
        make_trades([
            200,
            100,
            -50,
            -100,
        ]),
    )

    # gross profit = 300
    # gross loss = 150

    assert metrics.compute_profit_factor() == pytest.approx(2)


def test_profit_factor_only_winning_trades():
    metrics = MetricsComputer(
        make_history([100, 101]),
        make_trades([
            100,
            200,
        ]),
    )

    assert metrics.compute_profit_factor() == float("inf")


def test_profit_factor_no_trades():
    metrics = MetricsComputer(
        make_history([100, 101]),
        make_trades([]),
    )

    assert metrics.compute_profit_factor() == 0


def test_profit_factor_only_losses():
    metrics = MetricsComputer(
        make_history([100, 101]),
        make_trades([
            -100,
            -200,
        ]),
    )

    assert metrics.compute_profit_factor() == 0


# =========================================================
# VAR / CVAR
# =========================================================

def test_var_and_cvar():
    values = [100]

    for ret in [
        -0.10,
        -0.05,
        -0.02,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
    ]:
        values.append(
            values[-1] * (1 + ret)
        )

    metrics = MetricsComputer(
        make_history(values),
        make_trades([]),
    )

    var95, var99, cvar95, cvar99 = (
        metrics.compute_var_and_cvar()
    )

    expected_var95 = np.percentile(
        metrics.daily_ret,
        5,
    )

    expected_var99 = np.percentile(
        metrics.daily_ret,
        1,
    )

    assert var95 == pytest.approx(expected_var95)
    assert var99 == pytest.approx(expected_var99)

    assert cvar95 <= var95
    assert cvar99 <= var99


# =========================================================
# DRAWDOWN
# =========================================================

def test_no_drawdown():
    history = make_history([
        100,
        110,
        120,
        130,
    ])

    metrics = MetricsComputer(
        history,
        make_trades([]),
    )

    max_dd, max_dur, mean_dur = (
        metrics.compute_drawdown()
    )

    assert max_dd == 0
    assert max_dur == 0
    assert mean_dur == 0


def test_max_drawdown():
    history = make_history([
        100,
        120,
        90,
        120,
    ])

    metrics = MetricsComputer(
        history,
        make_trades([]),
    )

    max_dd, _, _ = metrics.compute_drawdown()

    # peak = 120
    # trough = 90
    #
    # drawdown = 90 / 120 - 1 = -25%

    assert max_dd == pytest.approx(-0.25)


def test_small_drawdown_is_ignored_for_recovery_average():
    history = make_history([
        100,
        99,
        100,
        110,
    ])

    metrics = MetricsComputer(
        history,
        make_trades([]),
    )

    _, _, mean_dur = metrics.compute_drawdown()

    # -1% is below the 2.5% noise threshold,
    # so it should not count as a drawdown episode.

    assert mean_dur == 0


# =========================================================
# RATIOS
# =========================================================

def test_ratios_with_constant_returns():
    history = make_history([
        100,
        110,
        121,
    ])

    metrics = MetricsComputer(
        history,
        make_trades([]),
    )

    sharpe, sortino, calmar = (
        metrics.compute_ratios(cagr=0.10)
    )

    assert np.isfinite(sharpe)
    assert np.isfinite(sortino)
    assert np.isfinite(calmar)


# =========================================================
# COMPUTE METRICS
# =========================================================

def test_compute_metrics_contains_expected_keys():
    history = make_history([
        100,
        102,
        101,
        105,
        110,
    ])

    trades = make_trades([
        100,
        -50,
    ])

    metrics = MetricsComputer(
        history,
        trades,
    )

    result = metrics.compute_metrics()

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

    assert set(result.keys()) == expected_keys