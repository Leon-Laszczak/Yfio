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
git clone https://github.com/Leon-Laszczak/Yfio
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

## Metrics

### Returns
- **Total PnL** Describes how much the strategy earned/lost
- **Percent PnL** Shows the total return of the strategy
- **CAGR** Indicates how much would the strategy return yearly on average
- **Volatility** Indicates how much the returns "swing" from the CAGR on average

### Risk Adjusted
- **Sharpe Ratio** Measures the risk-adjusted return of the strategy
- **Sortino Ratio** Measures strategy's return relative to its bad risk
- **Calmar Ratio** Measures strategy's return relative to its max drawdown

### Drawdown
- **Max Drawdown** Indicates the biggest decline from a local peak to the following low, before a new peak is reached
- **Max Recovery Time** Shows how long it took to be profitable from the max drawdown
- **Mean Recovery Time** Describes how long does it take on average for the strategy to become profitable again after a loss

### Tail Risk
- **VaR 1d 95%** Measures potential loss in the worst 5% of the days
- **VaR 1d 99%** Measures potential loss in the worst 1% of the days
- **CVaR 1d 95%** Describes average loss in the worst 5% of the days
- **CVaR 1d 99%** Describes average loss in the worst 1% of the days

### Trade Stats
- **Win Rate** Indicates what percent of the trades were profitable
- **Profit Factor** Shows how much money the strategy makes relative to its losses

### Distribution
- **Skewness** Shows if the returns of the data are stretched more to one side or the other
- **Kurtosis** Describes whether the tails of the returns distribution contain extreme values

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
├── main.py                  # Streamlit dashboard
├── requirements.txt
├── LICENSE
└── README.md
```

---

## License

This project is licensed under the [MIT License](LICENSE).