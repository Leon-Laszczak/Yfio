"""
data.py

Fetches stock prices history from Yahoo Finance.
"""
import yfinance as yf
import pandas as pd
from typing import Literal

PeriodType = Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
IntervalType = Literal["1m","2m","5m","15m","30m","60m","90m","1h","4h","1d","5d","1wk","1mo","3mo"]

def fetch_data(ticker : str, period : PeriodType = "1y", interval : IntervalType = "1d") -> pd.DataFrame:
    """ 
    Fetches stock prices history from yfinance.
    Attempts the provided ticker and a list of common Yahoo Finance
    suffixes until valid price data is found.

    Args:
        ticker: Stock symbol (can be without the suffix) eg. AAPL,TSLA,ASML.NV

        period: Time range of data to fetch. Supported values:
        1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y,
        ytd - year to date,
        max - full available history

        interval: Candle interval. Supported values:
        1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 4h, 1d, 5d, 1wk, 1mo, 3mo
        Please note that smaller intervals(below 1d) may have limited available history
    
    Returns:
        DataFrame with columns Open, High, Low, Close, Volume, indexed by date.
    
    Raises:
        ValueError if no data for the ticker is found

    Notes:
        If the requested period returns no data, the function retries with
        ``period="max"`` using the same interval. This can be useful when
        Yahoo Finance restricts the amount of historical data available
        for smaller intervals.

    Example:
        >>> df = download_data("AAPL","5y","1d")
        >>> print(df.head())
    """

    common_suffixes = ["",".L",".DE",".NV",".AS",".PA",".AX",".TO",".HK",".KS",".WA","=X","=F","-BTC"] # =X for forex pairs, =F for futures contracts, -BTC for cryptocurrencies
    base = ticker

    for suffix in common_suffixes:
        try:
            ticker = f"{base}{suffix}"
            data = yf.Ticker(ticker).history(period=period,interval=interval)

            if data.empty:
                # Retry with the maximum available history. Note that Yahoo Finance
                # still imposes history limits for smaller intervals.
                data = yf.Ticker(ticker).history(period="max", interval=interval) 
            required_columns = ["Open", "High", "Low", "Close", "Volume"]

            if not data.empty and all(col in data.columns for col in required_columns):
                data = data[~data.index.duplicated(keep="last")].sort_index()
                break

        except Exception:
            continue

    if data.empty or data is None:
        raise ValueError(f"No data found for ticker: {base}")
    
    return data[["Open","High","Low","Close","Volume"]]