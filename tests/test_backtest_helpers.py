from pathlib import Path
import sys

path = Path(__file__).parent.parent
sys.path.append(str(path.resolve()))

import pandas as pd
import pytest

from app.backtest.backtest import (
    calculate_value,
    close_trade,
    close_trades,
    execute_signal,
    get_exit_trigger,
    open_trade,
)


DATE = pd.Timestamp("2026-01-02")


class TestOpenTrade:
    def test_long_levels(self):
        trade = open_trade("LONG", 5.0, 100.0, DATE, tp=0.10, sl=0.05)

        assert trade["type"] == "LONG"
        assert trade["amount"] == pytest.approx(5.0)
        assert trade["entry_price"] == pytest.approx(100.0)
        assert trade["entry_date"] == DATE
        assert trade["take_profit"] == pytest.approx(110.0)
        assert trade["stop_loss"] == pytest.approx(95.0)

    def test_short_levels(self):
        trade = open_trade("SHORT", 5.0, 100.0, DATE, tp=0.10, sl=0.05)

        assert trade["take_profit"] == pytest.approx(90.0)
        assert trade["stop_loss"] == pytest.approx(105.0)

    def test_unknown_trade_type_raises(self):
        with pytest.raises(ValueError, match="Unknown trade type"):
            open_trade("SIDEWAYS", 1.0, 100.0, DATE, tp=0.10, sl=0.05)


class TestExitTrigger:
    @pytest.fixture
    def long_trade(self):
        return open_trade("LONG", 1.0, 100.0, DATE, tp=0.10, sl=0.05)

    @pytest.fixture
    def short_trade(self):
        return open_trade("SHORT", 1.0, 100.0, DATE, tp=0.10, sl=0.05)

    def test_long_stop_loss_intrabar(self, long_trade):
        assert get_exit_trigger(long_trade, 100, 104, 94) == pytest.approx((95.0, "SL"))

    def test_long_take_profit_intrabar(self, long_trade):
        assert get_exit_trigger(long_trade, 100, 111, 99) == pytest.approx((110.0, "TP"))

    def test_long_sl_has_priority_when_both_touched(self, long_trade):
        price, reason = get_exit_trigger(long_trade, 100, 111, 94)

        assert price == pytest.approx(95.0)
        assert reason == "SL"

    def test_long_gap_below_stop_fills_at_open(self, long_trade):
        price, reason = get_exit_trigger(long_trade, 90, 92, 88)

        assert price == pytest.approx(90.0)
        assert reason == "SL"

    def test_long_gap_above_tp_fills_at_open(self, long_trade):
        price, reason = get_exit_trigger(long_trade, 115, 117, 114)

        assert price == pytest.approx(115.0)
        assert reason == "TP"

    def test_short_stop_loss_intrabar(self, short_trade):
        price, reason = get_exit_trigger(short_trade, 100, 106, 98)

        assert price == pytest.approx(105.0)
        assert reason == "SL"

    def test_short_take_profit_intrabar(self, short_trade):
        price, reason = get_exit_trigger(short_trade, 100, 102, 89)

        assert price == pytest.approx(90.0)
        assert reason == "TP"

    def test_short_sl_has_priority_when_both_touched(self, short_trade):
        price, reason = get_exit_trigger(short_trade, 100, 106, 89)

        assert price == pytest.approx(105.0)
        assert reason == "SL"

    def test_short_gap_above_stop_fills_at_open(self, short_trade):
        price, reason = get_exit_trigger(short_trade, 110, 112, 108)

        assert price == pytest.approx(110.0)
        assert reason == "SL"

    def test_short_gap_below_tp_fills_at_open(self, short_trade):
        price, reason = get_exit_trigger(short_trade, 85, 87, 83)

        assert price == pytest.approx(85.0)
        assert reason == "TP"

    def test_no_trigger_returns_none(self, long_trade):
        assert get_exit_trigger(long_trade, 100, 105, 98) is None

    def test_unknown_type_raises(self):
        trade = {
            "type": "SIDEWAYS",
            "take_profit": 110.0,
            "stop_loss": 95.0,
        }

        with pytest.raises(ValueError, match="Unknown trade type"):
            get_exit_trigger(trade, 100, 105, 98)


class TestCloseTrade:
    def test_long_net_pnl_includes_both_fees(self):
        trade = open_trade("LONG", 10.0, 100.0, DATE, tp=0.10, sl=0.05)

        closed = close_trade(
            trade,
            exit_price=110.0,
            exit_date=pd.Timestamp("2026-01-03"),
            reason="TP",
            transaction_cost=0.001,
        )

        entry_fee = 10 * 100 * 0.001
        exit_fee = 10 * 110 * 0.001
        expected = 10 * (110 - 100) - entry_fee - exit_fee

        assert closed["pnl"] == pytest.approx(expected)
        assert closed["entry_fee"] == pytest.approx(entry_fee)
        assert closed["exit_fee"] == pytest.approx(exit_fee)
        assert closed["exit_reason"] == "TP"
        assert closed["take_profit"] == pytest.approx(110.0)
        assert closed["stop_loss"] == pytest.approx(95.0)

    def test_short_net_pnl_includes_both_fees(self):
        trade = open_trade("SHORT", 10.0, 100.0, DATE, tp=0.10, sl=0.05)

        closed = close_trade(
            trade,
            exit_price=90.0,
            exit_date=pd.Timestamp("2026-01-03"),
            reason="TP",
            transaction_cost=0.001,
        )

        entry_fee = 10 * 100 * 0.001
        exit_fee = 10 * 90 * 0.001
        expected = 10 * (100 - 90) - entry_fee - exit_fee

        assert closed["pnl"] == pytest.approx(expected)

    def test_flat_price_is_loss_equal_to_round_trip_fees(self):
        trade = open_trade("LONG", 2.0, 100.0, DATE, tp=0.10, sl=0.05)

        closed = close_trade(
            trade,
            exit_price=100.0,
            exit_date=pd.Timestamp("2026-01-03"),
            reason="FORCED_CLOSE",
            transaction_cost=0.001,
        )

        assert closed["pnl"] == pytest.approx(-0.4)


class TestCloseTrades:
    def test_long_cash_accounting(self):
        trade = open_trade("LONG", 10.0, 100.0, DATE, tp=0.10, sl=0.05)

        closed, cash = close_trades(
            [(trade, 110.0, "TP")],
            date=pd.Timestamp("2026-01-03"),
            transaction_cost=0.001,
            cash=9_000.0,
        )

        assert len(closed) == 1
        assert cash == pytest.approx(9_000 + 1_100 - 1.1)

    def test_short_cash_accounting(self):
        trade = open_trade("SHORT", 10.0, 100.0, DATE, tp=0.10, sl=0.05)

        closed, cash = close_trades(
            [(trade, 90.0, "TP")],
            date=pd.Timestamp("2026-01-03"),
            transaction_cost=0.001,
            cash=9_000.0,
        )

        expected = 9_000 + 1_000 + 100 - 0.9
        assert len(closed) == 1
        assert cash == pytest.approx(expected)

    def test_empty_collection_changes_nothing(self):
        closed, cash = close_trades(
            [],
            date=DATE,
            transaction_cost=0.001,
            cash=10_000.0,
        )

        assert closed == []
        assert cash == pytest.approx(10_000.0)


class TestExecuteSignal:
    def test_hold_is_noop(self):
        cash, trade = execute_signal(
            "HOLD", 100, 10_000, 0.001, DATE, 0.10, 0.05, 0.10, 10000
        )

        assert cash == pytest.approx(10_000)
        assert trade is None

    def test_buy_uses_requested_fraction_of_available_cash(self):
        cash, trade = execute_signal(
            "BUY", 100, 10_000, 0.001, DATE, 0.10, 0.05, 0.10, 10000
        )

        assert cash == pytest.approx(9_000)
        assert trade["type"] == "LONG"
        assert trade["amount"] * 100 * 1.001 == pytest.approx(1_000)

    def test_sell_opens_short_and_reserves_cash(self):
        cash, trade = execute_signal(
            "SELL", 100, 10_000, 0.001, DATE, 0.10, 0.05, 0.10, 10000
        )

        assert cash == pytest.approx(9_000)
        assert trade["type"] == "SHORT"

    def test_zero_cash_does_not_open_trade(self):
        cash, trade = execute_signal(
            "BUY", 100, 0, 0.001, DATE, 0.10, 0.05, 0.10, 10000
        )

        assert cash == 0
        assert trade is None

    def test_unknown_signal_raises(self):
        with pytest.raises(ValueError, match="Unknown signal"):
            execute_signal(
                "WAIT", 100, 10_000, 0.001, DATE, 0.10, 0.05, 0.10, 10000
            )


class TestCalculateValue:
    def test_long_mark_to_market(self):
        trade = open_trade("LONG", 10, 100, DATE, 0.10, 0.05)

        assert calculate_value(9_000, [trade], 110) == pytest.approx(10_100)

    def test_short_mark_to_market_profit_when_price_falls(self):
        trade = open_trade("SHORT", 10, 100, DATE, 0.10, 0.05)

        assert calculate_value(9_000, [trade], 90) == pytest.approx(10_100)

    def test_short_mark_to_market_loss_when_price_rises(self):
        trade = open_trade("SHORT", 10, 100, DATE, 0.10, 0.05)

        assert calculate_value(9_000, [trade], 110) == pytest.approx(9_900)

    def test_mixed_portfolio(self):
        long_trade = open_trade("LONG", 5, 100, DATE, 0.10, 0.05)
        short_trade = open_trade("SHORT", 5, 100, DATE, 0.10, 0.05)

        # At 110 the LONG is +50 and the SHORT is -50, so they offset.
        assert calculate_value(9_000, [long_trade, short_trade], 110) == pytest.approx(10_000)
