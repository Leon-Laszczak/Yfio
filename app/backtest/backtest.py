from app.strategy.strategy import strategy, MIN_LENGTH
from app.backtest.data import fetch_data
from app.backtest.metrics import MetricsComputer

import pandas as pd

def backtest(ticker: str, period: str, interval: str, transaction_cost: float = 0.001) -> tuple:
    """
    Backtesting engine.

    Args:
        ticker: Stock symbol (can be without the suffix) eg. AAPL,TSLA,ASML.NV

        period: Time range of data to fetch. Supported values:
        1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y,
        ytd - year to date,
        max - full available history
        
        interval: Candle interval. Supported values:
        1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 4h, 1d, 5d, 1wk, 1mo, 3mo
        Please note that smaller intervals(below 1d) may have limited available history

        transaction_cost: Cost of every transaction made in percent.
        Deafult 0.001 = 0.1% of the transaction. Used so the results are more realistic.
    
    Returns:
        A payload of metrics of the strategy portfolio.
    
    Raises:
        ValueError if a unknown signal or position type is passed

    Example:
        >>> payload = backtest("AAPL","1y","1d")
        >>> print(payload)
        {'Total PnL': np.float64(144.9443004790137), 'Percent PnL': np.float64(0.01449443004790137), 
        'CAGR': np.float64(0.03675605120363734), 'Voitality': np.float64(0.20789123039426688), 
        'Sharpe Ratio': np.float64(0.1768042410155026), 'Sortino Ratio': np.float64(0.2501154720271678), 
        'Calmar Ratio': np.float64(0.17338231895414866), 'Max Drawdown': -0.21199423000771817, 
        'Max Recovery Time': 170, 'Mean Recovery Time': 64.97570850202429, 'Var 1d 95%': np.float64(-0.02168260284769558),
        'Var 1d 99%': np.float64(-0.03920278867956136), 'CVar 1d 95%': np.float64(-0.03166088856080851), 
        'CVar 1d 99%': np.float64(-0.04653534479320979), 'Win Rate': 0.5555555555555556, 'Profit Factor': np.float64(-1.032350739470527),
        'Skewness': np.float64(-0.18816716761338828), 'Kurtosis': np.float64(2.3108627113436633)}

    """
    if transaction_cost < 0:
        raise ValueError("Transaction cost can't be nagative")
    cash = 10_000
    position_type = "FLAT"
    entry_price = 0
    entry_amount = 0
    trade_duration = 0

    df = fetch_data(ticker, period, interval)

    history = [10_000]
    trades = []

    for i in range(MIN_LENGTH, len(df) - 1):

        if position_type != "FLAT":
            trade_duration += 1

        historical_df = df.iloc[:i + 1]
        signal = strategy(historical_df)

        # trades are executed on next bar's open, not the signal bar's close
        price = df["Open"].iloc[i + 1]

        cash, position_type, entry_price, entry_amount, trade = execute_trade(
            signal, price, cash, position_type, entry_price, entry_amount, transaction_cost, df.index[:i+2], trade_duration
        )

        if trade is not None:
            trades.append(trade)
            trade_duration = 0

        curr_price = df["Close"].iloc[i + 1]
        value = calculate_value(cash, position_type, curr_price, entry_amount)
        history.append(value)

    last_price = df["Close"].iloc[-1]
    if position_type == "LONG":
        cash, position_type, entry_price, entry_amount, trade = close_long(last_price,cash,entry_price,entry_amount,transaction_cost,df.index[-1],df.index[-trade_duration-1])
        trades.append(trade)
        history[-1] = cash

    elif position_type == "SHORT":
        cash, position_type, entry_price, entry_amount, trade = close_short(last_price,cash,entry_price,entry_amount,transaction_cost,df.index[-1],df.index[-trade_duration-1])
        trades.append(trade)
        history[-1] = cash
        
    history = pd.Series(history)
    history.index = df.index[MIN_LENGTH:]

    if trades:
        trades = pd.DataFrame(trades)
    else:
        trades = pd.DataFrame({"type":[],"amount" : [], "entry_price":[],"close_price" : [], "pnl" : []})

    Metrics = MetricsComputer(history, trades)
    return Metrics.compute_metrics(), history, trades


def execute_trade(signal, price, cash, position_type, entry_price, entry_amount, transaction_cost, dates, trade_duration):
    """
    Executes a single trading decision based on the current signal and position state.

    No-op cases (BUY while LONG, SELL while SHORT, HOLD regardless of position)
    all just return the state unchanged, so they are collapsed into a single
    early-return instead of being duplicated per position type.
    """
    curr_date = dates[-1]
    open_date = dates[-trade_duration-1]

    # HOLD never changes anything, no matter the current position
    if signal == "HOLD":
        return cash, position_type, entry_price, entry_amount, None

    if signal == "BUY":
        # already long -> nothing to do (no pyramiding into the position)
        if position_type == "LONG":
            return cash, position_type, entry_price, entry_amount, None

        if position_type == "FLAT":
            return open_long(price, cash, transaction_cost)

        if position_type == "SHORT":
            return close_short(price, cash, entry_price, entry_amount, transaction_cost, curr_date, open_date)

    if signal == "SELL":
        if position_type == "LONG":
            return close_long(price, cash, entry_price, entry_amount, transaction_cost, curr_date, open_date)

        if position_type == "FLAT":
            return open_short(price, cash, transaction_cost)

        # already short -> nothing to do (no pyramiding into the position)
        if position_type == "SHORT":
            return cash, position_type, entry_price, entry_amount, None

    raise ValueError(f"Unknown signal/position combination: {signal}, {position_type}")


def open_long(price, cash, transaction_cost):
    """Spend all cash to open a LONG position."""
    amount = cash / (price * (1 + transaction_cost))
    cash = 0
    return cash, "LONG", price, amount, None


def close_long(price, cash, entry_price, entry_amount, transaction_cost,close_date, open_date):
    """Sell out of a LONG position and realize PnL."""
    pnl = (
        entry_amount * price * (1 - transaction_cost)
        - entry_amount * entry_price * (1 + transaction_cost)
    )

    trade = {
        "open_date" : open_date,
        "close_date" : close_date, 
        "type": "LONG",
        "amount": entry_amount,
        "entry_price": entry_price,
        "close_price": price,
        "pnl": pnl,
    }

    cash += entry_amount * price * (1 - transaction_cost)
    return cash, "FLAT", 0, 0, trade


def open_short(price, cash, transaction_cost):
    """Sell borrowed shares (sized against current cash) to open a SHORT position."""
    amount = cash / (price * (1 + transaction_cost))
    cash += amount * price * (1 - transaction_cost)
    return cash, "SHORT", price, amount, None


def close_short(price, cash, entry_price, entry_amount, transaction_cost, close_date, open_date):
    """Buy back a SHORT position and realize PnL."""
    pnl = (
        entry_amount * entry_price * (1 - transaction_cost)
        - entry_amount * price * (1 + transaction_cost)
    )

    trade = {
        "open_date" : open_date,
        "close_date" : close_date, 
        "type": "SHORT",
        "amount": entry_amount,
        "entry_price": entry_price,
        "close_price": price,
        "pnl": pnl,
    }

    cash -= entry_amount * price * (1 + transaction_cost)
    return cash, "FLAT", 0, 0, trade


def calculate_value(cash, position_type, price, entry_amount):
    """Mark-to-market current portfolio value given the open position."""
    if position_type == "FLAT":
        return cash
    elif position_type == "LONG":
        return cash + entry_amount * price
    elif position_type == "SHORT":
        return cash - entry_amount * price
    else:
        raise ValueError(f"Unknown position type: {position_type}")