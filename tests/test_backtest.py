from pathlib import Path
import sys

path = Path(__file__).parent.parent
sys.path.append(str(path.resolve()))

import pytest

from app.backtest.backtest import (
    execute_trade,
    open_long,
    close_long,
    open_short,
    close_short,
    calculate_value,
)


# =========================================================
# OPEN LONG
# =========================================================

def test_open_long_without_transaction_cost():
    cash, position, entry_price, amount, trade = (
        open_long(
            price=100,
            cash=10_000,
            transaction_cost=0,
        )
    )

    assert cash == 0
    assert position == "LONG"
    assert entry_price == 100
    assert amount == 100
    assert trade is None


def test_open_long_with_transaction_cost():
    cash, position, entry_price, amount, trade = (
        open_long(
            price=100,
            cash=10_000,
            transaction_cost=0.001,
        )
    )

    expected_amount = (
        10_000 / (100 * 1.001)
    )

    assert cash == 0
    assert position == "LONG"
    assert entry_price == 100
    assert amount == pytest.approx(expected_amount)
    assert trade is None


# =========================================================
# CLOSE LONG
# =========================================================

def test_close_long_profit():
    cash, position, entry_price, amount, trade = (
        close_long(
            price=120,
            cash=0,
            entry_price=100,
            entry_amount=10,
            transaction_cost=0,
            close_date="close",
            open_date="open",
        )
    )

    assert cash == 1200
    assert position == "FLAT"
    assert entry_price == 0
    assert amount == 0

    assert trade["type"] == "LONG"
    assert trade["pnl"] == 200
    assert trade["open_date"] == "open"
    assert trade["close_date"] == "close"


def test_close_long_loss():
    _, _, _, _, trade = close_long(
        price=80,
        cash=0,
        entry_price=100,
        entry_amount=10,
        transaction_cost=0,
        close_date="close",
        open_date="open",
    )

    assert trade["pnl"] == -200


# =========================================================
# OPEN SHORT
# =========================================================

def test_open_short_without_transaction_cost():
    cash, position, entry_price, amount, trade = (
        open_short(
            price=100,
            cash=10_000,
            transaction_cost=0,
        )
    )

    assert cash == 20_000
    assert position == "SHORT"
    assert entry_price == 100
    assert amount == 100
    assert trade is None


# =========================================================
# CLOSE SHORT
# =========================================================

def test_close_short_profit():
    cash, position, entry_price, amount, trade = (
        close_short(
            price=80,
            cash=20_000,
            entry_price=100,
            entry_amount=100,
            transaction_cost=0,
            close_date="close",
            open_date="open",
        )
    )

    assert cash == 12_000
    assert position == "FLAT"
    assert entry_price == 0
    assert amount == 0

    assert trade["type"] == "SHORT"
    assert trade["pnl"] == 2000


def test_close_short_loss():
    _, _, _, _, trade = close_short(
        price=120,
        cash=20_000,
        entry_price=100,
        entry_amount=100,
        transaction_cost=0,
        close_date="close",
        open_date="open",
    )

    assert trade["pnl"] == -2000


# =========================================================
# PORTFOLIO VALUE
# =========================================================

def test_calculate_value_flat():
    assert calculate_value(
        cash=10_000,
        position_type="FLAT",
        price=100,
        entry_amount=0,
    ) == 10_000


def test_calculate_value_long():
    assert calculate_value(
        cash=0,
        position_type="LONG",
        price=110,
        entry_amount=100,
    ) == 11_000


def test_calculate_value_short():
    assert calculate_value(
        cash=20_000,
        position_type="SHORT",
        price=90,
        entry_amount=100,
    ) == 11_000


def test_calculate_value_invalid_position():
    with pytest.raises(ValueError):
        calculate_value(
            cash=10_000,
            position_type="INVALID",
            price=100,
            entry_amount=0,
        )


# =========================================================
# EXECUTE TRADE
# =========================================================

def test_hold_does_nothing():
    result = execute_trade(
        signal="HOLD",
        price=100,
        cash=10_000,
        position_type="FLAT",
        entry_price=0,
        entry_amount=0,
        transaction_cost=0,
        dates=[1],
        trade_duration=0,
    )

    assert result == (
        10_000,
        "FLAT",
        0,
        0,
        None,
    )


def test_buy_when_flat_opens_long():
    cash, position, entry_price, amount, trade = (
        execute_trade(
            signal="BUY",
            price=100,
            cash=10_000,
            position_type="FLAT",
            entry_price=0,
            entry_amount=0,
            transaction_cost=0,
            dates=[1],
            trade_duration=0,
        )
    )

    assert cash == 0
    assert position == "LONG"
    assert entry_price == 100
    assert amount == 100
    assert trade is None


def test_sell_when_flat_opens_short():
    cash, position, entry_price, amount, trade = (
        execute_trade(
            signal="SELL",
            price=100,
            cash=10_000,
            position_type="FLAT",
            entry_price=0,
            entry_amount=0,
            transaction_cost=0,
            dates=[1],
            trade_duration=0,
        )
    )

    assert cash == 20_000
    assert position == "SHORT"
    assert entry_price == 100
    assert amount == 100
    assert trade is None


def test_buy_when_already_long_does_nothing():
    result = execute_trade(
        signal="BUY",
        price=120,
        cash=0,
        position_type="LONG",
        entry_price=100,
        entry_amount=100,
        transaction_cost=0,
        dates=[1],
        trade_duration=0,
    )

    assert result == (
        0,
        "LONG",
        100,
        100,
        None,
    )


def test_sell_when_already_short_does_nothing():
    result = execute_trade(
        signal="SELL",
        price=80,
        cash=20_000,
        position_type="SHORT",
        entry_price=100,
        entry_amount=100,
        transaction_cost=0,
        dates=[1],
        trade_duration=0,
    )

    assert result == (
        20_000,
        "SHORT",
        100,
        100,
        None,
    )


def test_invalid_signal_raises_value_error():
    with pytest.raises(ValueError):
        execute_trade(
            signal="INVALID",
            price=100,
            cash=10_000,
            position_type="FLAT",
            entry_price=0,
            entry_amount=0,
            transaction_cost=0,
            dates=[1],
            trade_duration=0,
        )