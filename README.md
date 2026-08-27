# Yfio

Yfio is a backtesting engine for hobby traders. It helps you check if the strategy has any historical sense before you risk your own money. Strategies are fully pluggable — drop your logic into `app/strategy/strategy.py` and the engine will take care of the rest by simulating trades on historical data.

---

## Features

- Fetching historical OHLCV data from Yahoo Finance with automatic suffix detection in case the whole ticker isn't specified
- Strategy tester that allows you to open LONG and SHORT positions with transaction costs so the results are not overstated
- Metrics computer that calculates a rich set of performance and risk metrics — Sharpe/Sortino/Calmar ratios, VaR/CVaR, drawdown analysis, win rate, and more
- Pluggable strategy interface that allows you to implement your own logic without touching the engine
- Interactive Streamlit dashboard to configure your backtest run and get insights on metrics and performance of your strategy

---

## Getting Started

### Installation

```bash
git clone https://github.com/Leon-Laszczak/Yfio.git
cd Yfio
pip install -r requirements.txt
```

### How to Use

After you install, plug in your trading strategy into `app/strategy/strategy.py` and then run

```bash
streamlit run main.py
```

After that you can open `http://localhost:8501`, enter the ticker and customize period, interval and transaction cost, and test your strategy.

---

## Testing

Yfio includes unit and integration tests for the backtesting engine, metrics, trade execution, and edge cases.

Run all tests with:

```bash
pytest -v
```

The test suite covers, among other things:

- LONG and SHORT position execution
- Transaction costs
- Portfolio valuation
- Forced position closing at the end of a backtest
- Strategy minimum history length
- PnL and win rate calculations
- Profit factor
- VaR and CVaR
- Drawdowns and recovery times
- Full end-to-end backtest scenarios

---

## Writing a Strategy

Strategies live in `app/strategy/strategy.py`.

A strategy receives historical OHLCV data up to the current candle and should return one of three signals:

- `BUY`
- `HOLD`
- `SELL`

Example:

```python
MIN_LENGTH = 20

def strategy(df):
    short_ma = df["Close"].rolling(5).mean().iloc[-1]
    long_ma = df["Close"].rolling(20).mean().iloc[-1]

    if short_ma > long_ma:
        return "BUY"

    if short_ma < long_ma:
        return "SELL"

    return "HOLD"
```

MIN_LENGTH defines how many candles the strategy needs before the backtest begins.

Trades are executed on the next candle's Open price to reduce look-ahead bias.
---

## Metrics

### Returns
- **Total PnL** Describes how much the strategy earned/lost
- **Percent PnL** Shows the total return of the strategy
- **CAGR** Indicates how much would the strategy return yearly on average
- **Volatility** Measures the annualized variability of strategy returns

### Risk Adjusted
- **Sharpe Ratio** Measures the risk-adjusted return of the strategy
- **Sortino Ratio** Measures strategy's return relative to its bad risk
- **Calmar Ratio** Measures strategy's return relative to its max drawdown

### Drawdown
- **Max Drawdown** Indicates the biggest decline from a local peak to the following low, before a new peak is reached
- **Max Recovery Time** Shows the longest time required for the portfolio to recover from a drawdown and reach its previous peak
- **Mean Recovery Time** Shows the average duration of significant drawdown episodes before recovery

### Tail Risk
- **VaR 1d 95%** Estimates the daily return threshold exceeded by losses on approximately 5% of days
- **VaR 1d 99%** Estimates the daily return threshold exceeded by losses on approximately 1% of days
- **CVaR 1d 95%** Measures the average daily return on days worse than the 95% VaR threshold
- **CVaR 1d 99%** Measures the average daily return on days worse than the 99% VaR threshold

### Trade Stats
- **Win Rate** Indicates what percent of the trades were profitable
- **Profit Factor** Shows how much money the strategy makes relative to its losses

### Distribution
- **Skewness** Shows if the returns of the data are stretched more to one side or the other
- **Kurtosis** Describes whether the tails of the returns distribution contain extreme values

---

## Current Limitations

Yfio is currently an early-stage backtesting engine intended primarily for learning, experimentation, and hobby use.

Current limitations include:

- One open position at a time
- No take-profit or stop-loss orders yet
- No configurable position sizing yet
- No portfolio-level multi-asset backtesting
- No slippage model
- Historical data depends on Yahoo Finance availability and accuracy

---

## Roadmap

Planned features include:

- [ ] Take-profit and stop-loss orders
- [ ] Configurable position sizing
- [ ] Risk-based position sizing
- [ ] Improved trade management
- [ ] Slippage modeling
- [ ] Multi-asset portfolio backtesting
- [ ] Projection of future strategy returns
- [ ] More strategy examples

---

## Project Structure
```
Yfio/
├── app/
│   ├── backtest/
│   │   ├── backtest.py      # Trade simulation engine
│   │   ├── data.py          # Historical data fetching
│   │   └── metrics.py       # Performance & risk metrics
│   └── strategy/
│       └── strategy.py      # Your custom strategy goes here
├── tests/
│   ├── test_backtest.py
│   ├── test_backtest_integration.py
│   └── test_metrics.py
├── main.py                  # Streamlit dashboard
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Disclaimer

Yfio is intended for educational and research purposes only.

Backtest results do not guarantee future performance. Historical data, transaction costs, execution assumptions, and simplified market mechanics may differ from real trading conditions.

This project does not provide financial or investment advice.

---
## License

This project is licensed under the [MIT License](LICENSE).