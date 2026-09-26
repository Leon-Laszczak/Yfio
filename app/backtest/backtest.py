from app.strategy.strategy import strategy, MIN_LENGTH
from app.backtest.data import fetch_data
from app.backtest.metrics import MetricsComputer

import pandas as pd
import math


INITIAL_CASH = 10_000.0

TRADE_COLUMNS = [
    "type",
    "amount",
    "entry_price",
    "close_price",
    "entry_date",
    "close_date",
    "take_profit",
    "stop_loss",
    "entry_fee",
    "exit_fee",
    "pnl",
    "exit_reason",
]


def backtest(
    ticker: str,
    period: str,
    interval: str,
    transaction_cost: float = 0.001,
    position_size: float = 0.1,
    position_size_base: str = "equity",
    take_profit: float = 0.1,
    stop_loss: float = 0.05,
) -> tuple[dict, pd.Series, pd.DataFrame]:
    """
    Run a multi-position backtest for a single asset.

    The strategy is evaluated using data available through the current candle.
    Its signal is executed on the next candle's Open, preventing execution on
    the same Close that produced the signal.

    Signal semantics:
        BUY:
            Open a new LONG trade.
        SELL:
            Open a new SHORT trade.
        HOLD:
            Do not open a new trade.

    Existing trades are never closed by later strategy signals. Each trade is
    closed only when its own take-profit or stop-loss is reached, or at the
    final Close of the backtest.

    Multiple LONG and SHORT trades may therefore be open at the same time.

    Execution assumptions:
        - Signals are executed on the next candle's Open.
        - Each new trade uses ``position_size`` of the selected sizing base,
          including its entry fee. The engine skips unaffordable entries.
        - Entry sizing and execution precede TP/SL exits on the same candle.
        - LONG and SHORT trades both reserve their position value from cash.
          SHORT trades therefore use a simplified collateral model rather than
          broker-specific margin rules.
        - No margin calls, borrow fees, interest, slippage, partial fills, or
          liquidity constraints are modeled.
        - TP/SL levels are checked using the candle's Open, High, and Low.
        - If price gaps through TP/SL, the trade is filled at the candle Open.
        - If both TP and SL are touched intrabar and their order is unknowable
          from OHLC data, SL is assumed to occur first (conservative rule).
        - Any positions still open at the end are closed at the final Close.
        - Transaction costs are charged on both entry and exit.

    Args:
        ticker:
            Stock symbol accepted by ``fetch_data``. It may be supplied without
            a market suffix, for example ``AAPL``, ``TSLA``, or ``ASML.NV``.

        period:
            Time range passed to ``fetch_data``. Supported values include:
            ``1d``, ``5d``, ``1mo``, ``3mo``, ``6mo``, ``1y``, ``2y``,
            ``5y``, ``10y``, ``ytd``, and ``max``.

        interval:
            Candle interval passed to ``fetch_data``. Supported values include:
            ``1m``, ``2m``, ``5m``, ``15m``, ``30m``, ``60m``, ``90m``,
            ``1h``, ``4h``, ``1d``, ``5d``, ``1wk``, ``1mo``, and ``3mo``.
            Intraday intervals may have limited historical availability.

        transaction_cost:
            Proportional cost charged on each entry and exit transaction.
            Default ``0.001`` means 0.1% per transaction.

        position_size:
            Fraction of ``position_size_base`` allocated to a new trade,
            including its entry fee. Must be in ``(0, 1]``. Default ``0.1``
            requests 10% of the selected base, subject to available cash.

        position_size_base:
            ``initial`` uses INITIAL_CASH, ``equity`` uses portfolio value at
            the execution candle's Open, and ``cash`` uses available cash.
            The function defaults to ``equity``. Existing positions affect
            equity through their value at that Open, before any TP/SL exits.

        take_profit:
            Fractional take-profit distance from entry. Must be positive.
            Default ``0.1`` means 10%.
            LONG TP = entry * (1 + take_profit)
            SHORT TP = entry * (1 - take_profit)

        stop_loss:
            Fractional stop-loss distance from entry. Must be positive.
            Default ``0.05`` means 5%.
            LONG SL = entry * (1 - stop_loss)
            SHORT SL = entry * (1 + stop_loss)

    Returns:
        tuple[dict, pandas.Series, pandas.DataFrame]:
            metrics:
                Portfolio metrics returned by ``MetricsComputer``.
            history:
                Mark-to-market portfolio equity over the backtest.
            trades:
                One row per closed trade. Includes entry/exit prices and dates,
                TP/SL levels, transaction fees, PnL, and exit reason.

    Raises:
        ValueError:
            If transaction costs, position size, or TP/SL inputs are invalid,
            history is too short, or a sizing-base value encountered in the
            loop is unknown. Data loading and metric computation can also fail.

    Notes:
        The entry handler catches all ValueError exceptions from execute_signal
        and skips that entry, including an unsupported strategy signal. It does
        not distinguish insufficient cash from other entry validation errors.

    Example:
        >>> metrics, history, trades = backtest(
        ...     ticker="AAPL",
        ...     period="1y",
        ...     interval="1d",
        ...     transaction_cost=0.001,
        ...     position_size=0.1,
        ...     take_profit=0.1,
        ...     stop_loss=0.05,
        ... )
        >>> print(metrics)
        >>> print(trades.tail())
    """
    if not 0 <= transaction_cost <= 1:
        raise ValueError("transaction_cost must be in the interval [0, 1]")

    if not 0 < position_size <= 1:
        raise ValueError("position_size must be in the interval (0, 1]")

    if take_profit <= 0:
        raise ValueError("take_profit must be positive")

    if  not 0 < stop_loss <= 1:
        raise ValueError("stop_loss must be in the interval (0, 1]")

    df = fetch_data(ticker, period, interval)

    if len(df) <= MIN_LENGTH:
        raise ValueError(
            f"Not enough data to run the strategy: got {len(df)} rows, "
            f"but more than MIN_LENGTH={MIN_LENGTH} rows are required"
        )

    cash = INITIAL_CASH
    open_trades: list[dict] = []
    closed_trades: list[dict] = []

    # The first history point is the untouched portfolio value at the first
    # candle on which the strategy has enough warm-up data.
    history = [INITIAL_CASH]

    for i in range(MIN_LENGTH, len(df) - 1):
        historical_df = df.iloc[: i + 1]
        signal = strategy(historical_df)

        if signal not in ["BUY","SELL","HOLD"]:
            raise ValueError("Unknown signal was passed to the backtesting engine")

        execution_date = df.index[i + 1]
        open_price = float(df["Open"].iloc[i + 1])
        high = float(df["High"].iloc[i + 1])
        low = float(df["Low"].iloc[i + 1])
        close_price = float(df["Close"].iloc[i + 1])

        equity_at_open = calculate_value(
            cash,
            open_trades,
            open_price,
        )

        if position_size_base == "initial":
            sizing_base = INITIAL_CASH

        elif position_size_base == "equity":
            sizing_base = equity_at_open

        elif position_size_base == "cash":
            sizing_base = cash
        else:
            raise ValueError(
                "position_size_base must be 'initial', 'equity', or 'cash'"
            )
        
        # 1. Execute the signal from candle i on candle i+1 Open.
        try:
            cash, new_trade = execute_signal(
                signal=signal,
                price=open_price,
                cash=cash,
                transaction_cost=transaction_cost,
                date=execution_date,
                tp=take_profit,
                sl=stop_loss,
                position_size=position_size,
                sizing_base = sizing_base
            )
        except ValueError:
            # Skip rejected entries, including insufficient cash or an invalid signal.
            new_trade = None

        if new_trade is not None:
            open_trades.append(new_trade)

        # 2. Check every open trade for TP/SL on the execution candle.
        trades_to_close = []

        for trade in open_trades:
            trigger = get_exit_trigger(
                trade=trade,
                open_price=open_price,
                high=high,
                low=low,
            )

            if trigger is not None:
                exit_price, reason = trigger
                trades_to_close.append((trade, exit_price, reason))

        # 3. Close triggered trades exactly once and remove them from open trades.
        newly_closed, cash = close_trades(
            trades_to_close=trades_to_close,
            date=execution_date,
            transaction_cost=transaction_cost,
            cash=cash,
        )
        closed_trades.extend(newly_closed)

        for trade, _, _ in trades_to_close:
            open_trades.remove(trade)

        # 4. Mark remaining positions to market at this candle's Close.
        value = calculate_value(
            cash=cash,
            trades=open_trades,
            price=close_price,
        )
        history.append(value)

    # 5. Force-close every position still open at the final Close.
    if open_trades:
        final_date = df.index[-1]
        final_price = float(df["Close"].iloc[-1])

        forced_closes = [
            (trade, final_price, "FORCED_CLOSE")
            for trade in open_trades
        ]

        newly_closed, cash = close_trades(
            trades_to_close=forced_closes,
            date=final_date,
            transaction_cost=transaction_cost,
            cash=cash,
        )
        closed_trades.extend(newly_closed)
        open_trades.clear()

        # The final history point must include exit costs from forced closes.
        history[-1] = cash

    history = pd.Series(
        history,
        index=df.index[MIN_LENGTH:],
        dtype=float,
        name="equity",
    )

    trades = pd.DataFrame(closed_trades, columns=TRADE_COLUMNS)

    metrics = MetricsComputer(history, trades)
    return metrics.compute_metrics(), history, trades

def greater_or_equal(a: float, b: float) -> bool:
    return a > b or math.isclose(
        a,
        b,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def open_trade(
    trade_type: str,
    entry_amount: float,
    entry_price: float,
    entry_date,
    tp: float,
    sl: float,
) -> dict:
    """
    Create a new LONG or SHORT trade with fixed TP and SL price levels.
    """
    if trade_type == "LONG":
        tp_price = entry_price * (1 + tp)
        sl_price = entry_price * (1 - sl)

    elif trade_type == "SHORT":
        tp_price = entry_price * (1 - tp)
        sl_price = entry_price * (1 + sl)

    else:
        raise ValueError(f"Unknown trade type: {trade_type}")

    return {
        "type": trade_type,
        "entry_date": entry_date,
        "amount": entry_amount,
        "entry_price": entry_price,
        "take_profit": tp_price,
        "stop_loss": sl_price,
    }


def get_exit_trigger(
    trade: dict,
    open_price: float,
    high: float,
    low: float,
) -> tuple[float, str] | None:
    """
    Return the exit price and reason if a trade hits TP/SL on a candle.

    Gaps are filled at the candle Open. If both TP and SL are reachable within
    the same OHLC candle and their order cannot be determined, SL has priority.
    """
    if trade["type"] == "LONG":
        if greater_or_equal(trade["stop_loss"],open_price):
            return open_price, "SL"

        if greater_or_equal(open_price,trade["take_profit"]):
            return open_price, "TP"

        if greater_or_equal(trade["stop_loss"],low):
            return trade["stop_loss"], "SL"

        if greater_or_equal(high,trade["take_profit"]):
            return trade["take_profit"], "TP"

    elif trade["type"] == "SHORT":
        if greater_or_equal(open_price,trade["stop_loss"]):
            return open_price, "SL"

        if greater_or_equal(trade["take_profit"],open_price):
            return open_price, "TP"

        if greater_or_equal(high,trade["stop_loss"]):
            return trade["stop_loss"], "SL"

        if greater_or_equal(trade["take_profit"],low):
            return trade["take_profit"], "TP"

    else:
        raise ValueError(f"Unknown trade type: {trade['type']}")

    return None


def close_trade(
    trade: dict,
    exit_price: float,
    exit_date,
    reason: str,
    transaction_cost: float,
) -> dict:
    """
    Convert an open trade into a closed-trade record and calculate net PnL.
    """
    amount = trade["amount"]
    entry_price = trade["entry_price"]

    entry_fee = amount * entry_price * transaction_cost
    exit_fee = amount * exit_price * transaction_cost

    if trade["type"] == "LONG":
        gross_pnl = amount * (exit_price - entry_price)

    elif trade["type"] == "SHORT":
        gross_pnl = amount * (entry_price - exit_price)

    else:
        raise ValueError(f"Unknown trade type: {trade['type']}")

    pnl = gross_pnl - entry_fee - exit_fee

    return {
        "type": trade["type"],
        "amount": amount,
        "entry_price": entry_price,
        "close_price": exit_price,
        "entry_date": trade["entry_date"],
        "close_date": exit_date,
        "take_profit": trade["take_profit"],
        "stop_loss": trade["stop_loss"],
        "entry_fee": entry_fee,
        "exit_fee": exit_fee,
        "pnl": pnl,
        "exit_reason": reason,
    }


def close_trades(
    trades_to_close: list[tuple[dict, float, str]],
    date,
    transaction_cost: float,
    cash: float,
) -> tuple[list[dict], float]:
    """
    Close a collection of trades and return their records plus updated cash.

    LONG positions return their sale proceeds minus the exit fee.

    SHORT positions use the simplified collateral model used by this engine:
    the initially reserved position value is returned together with the gross
    short PnL, then the exit fee is deducted. The entry fee was already paid
    when the trade was opened.
    """
    closed = []

    for trade, exit_price, reason in trades_to_close:
        closed_trade = close_trade(
            trade=trade,
            exit_price=exit_price,
            exit_date=date,
            reason=reason,
            transaction_cost=transaction_cost,
        )

        amount = trade["amount"]
        entry_value = amount * trade["entry_price"]
        exit_value = amount * exit_price
        exit_fee = exit_value * transaction_cost

        if trade["type"] == "LONG":
            cash += exit_value - exit_fee

        elif trade["type"] == "SHORT":
            gross_pnl = amount * (trade["entry_price"] - exit_price)
            cash += entry_value + gross_pnl - exit_fee

        else:
            raise ValueError(f"Unknown trade type: {trade['type']}")

        closed.append(closed_trade)

    return closed, cash


def execute_signal(
    signal: str,
    price: float,
    cash: float,
    transaction_cost: float,
    date,
    tp: float,
    sl: float,
    position_size: float,
    sizing_base: float,
) -> tuple[float, dict | None]:
    """Execute a strategy signal at the current candle's Open.

    BUY requests a new LONG trade and SELL requests a new SHORT trade.
    HOLD or non-positive available cash returns without opening a trade.
    Existing positions remain under TP/SL and final-close management.

    The caller supplies sizing_base from initial capital, current equity, or
    available cash. The new position consumes sizing_base * position_size,
    including its entry fee, subject to floating-point precision.

    Returns:
        Updated cash and the new trade, or unchanged cash and None when no
        trade is requested or cash is non-positive.

    Raises:
        ValueError: With positive cash, an unsupported signal or an entry cost
            above available cash rejects the entry. The backtest caller catches
            these errors and continues without opening that trade.
    """
    if signal == "HOLD":
        return cash, None

    if cash <= 0:
        return cash, None

    if signal not in {"BUY", "SELL"}:
        raise ValueError(f"Unknown signal: {signal}")

    amount = (
        sizing_base * position_size
        / (price * (1 + transaction_cost))
    )

    position_value = amount * price
    entry_fee = position_value * transaction_cost

    if position_value + entry_fee > cash:
        raise ValueError(
            f"Not enough cash to open a new {signal} position: "
            f"cash={cash}, required={position_value + entry_fee}"
        )
    
    cash -= position_value + entry_fee

    trade_type = "LONG" if signal == "BUY" else "SHORT"

    trade = open_trade(
        trade_type=trade_type,
        entry_amount=amount,
        entry_price=price,
        entry_date=date,
        tp=tp,
        sl=sl,
    )

    return cash, trade


def calculate_value(
    cash: float,
    trades: list[dict],
    price: float,
) -> float:
    """
    Calculate mark-to-market portfolio equity at a given price.

    Entry fees have already been deducted from cash. Potential future exit fees
    are not deducted until a trade is actually closed.
    """
    value = cash

    for trade in trades:
        entry_value = trade["amount"] * trade["entry_price"]

        if trade["type"] == "LONG":
            current_value = trade["amount"] * price

        elif trade["type"] == "SHORT":
            unrealized_pnl = trade["amount"] * (
                trade["entry_price"] - price
            )
            current_value = entry_value + unrealized_pnl

        else:
            raise ValueError(f"Unknown trade type: {trade['type']}")

        value += current_value

    return value
