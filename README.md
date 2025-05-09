# ClaudeCTA: Momentum-Based Trading Strategies Framework

A comprehensive framework for researching, implementing, and backtesting momentum-based trading strategies, from the classic Turtle Trading system to modern quantitative approaches.

## Overview

ClaudeCTA is a Python framework designed for quant traders and researchers interested in momentum-based trading strategies. The project includes:

1. Classic Turtle Trading System implementation
2. Modern momentum strategies (time-series, cross-sectional, hybrid)
3. Advanced risk management techniques
4. Comprehensive backtesting engine
5. Performance analytics and visualization tools

## Momentum Trading: Strategy Background

### The Turtle Trading System

The Turtle Trading system originated in 1983 when commodities traders Richard Dennis and William Eckhardt conducted an experiment to determine whether successful trading could be taught. Dennis recruited novice traders (dubbed "Turtles"), trained them for two weeks, and provided them with trading accounts. Over five years, these traders collectively earned approximately $175 million, proving that trading could indeed be taught.

#### Key Turtle Trading Rules:

1. **Entry System**:
   - System 1: Enter on 20-day breakouts (short-term)
   - System 2: Enter on 55-day breakouts (long-term)

2. **Position Sizing**:
   - Uses the Average True Range (N) to determine volatility
   - Risk is standardized across markets: 1% of account equity per trade
   - Position size = (1% × Account) ÷ (N × Dollar value per point)

3. **Risk Management**:
   - 2N stop loss from entry point
   - Maximum units per market: 4
   - Maximum units per correlated markets: 6
   - Maximum units per loosely correlated markets: 10
   - Maximum total portfolio exposure: 20 units

4. **Exit Rules**:
   - System 1: Exit on 10-day opposite breakout
   - System 2: Exit on 20-day opposite breakout
   - Pyramiding: Add units at 1/2N intervals up to maximum allocation

### Evolution to Modern Momentum Strategies

Modern momentum strategies have evolved significantly from the original Turtle Trading system. Key advancements include:

#### Types of Modern Momentum Strategies:

1. **Time-Series Momentum**:
   - Based on absolute performance of individual assets
   - Buy when an asset has positive returns over lookback period; sell when negative
   - Most effective during trending markets

2. **Cross-Sectional Momentum**:
   - Based on relative performance between assets within a universe
   - Long top performers, short bottom performers
   - Provides more consistent exposure across market conditions

3. **Hybrid Approaches**:
   - Combines elements of both time-series and cross-sectional methods
   - Adapts to different market conditions
   - Delivers more stable risk-adjusted returns

#### Key Modern Enhancements:

1. **Volatility Scaling**:
   - Scale positions inversely to volatility
   - Target constant volatility (typically 10-12%)
   - Nearly doubles Sharpe ratio
   - Virtually eliminates catastrophic momentum crashes

2. **Dynamic Risk Management**:
   - Adjusts exposure based on forecasts of momentum's mean and variance
   - Reduces exposure during "panic states" (high volatility after market declines)
   - Enhances returns during market rebounds

3. **Factor Timing and Regime Filters**:
   - Uses market regimes to determine when to apply momentum
   - Implements inverse strategy during bear markets
   - Applies filters based on macro indicators, volatility, and sentiment

4. **Machine Learning Enhancements**:
   - Improves signal generation and reduces noise
   - Incorporates additional data sources beyond price
   - Identifies complex non-linear patterns in market behavior
   - Adapts to changing market conditions

## Project Components

### Strategies

- **Turtle Trading System**
  - Original System 1 (20-day breakout)
  - Original System 2 (55-day breakout)

- **Modern Momentum Strategies**
  - Time-Series Momentum
  - Cross-Sectional Momentum
  - Adaptive Momentum (regime-aware)

### Risk Management

- **Position Sizing**
  - Fixed Dollar
  - Fixed Risk (percentage)
  - ATR-based (Turtle-style)
  - Volatility-scaled

- **Volatility Scaling**
  - Constant Volatility Targeting
  - Dynamic Volatility Targeting
  - Adaptive Volatility Targeting
  - Regime-based Adjustments

### Backtesting Engine

- Event-driven architecture
- Multiple asset class support
- Transaction cost modeling
- Realistic simulation of position sizing and risk management
- Comprehensive performance metrics

### Performance Analysis

- Return metrics (total, annualized, etc.)
- Risk metrics (volatility, drawdown, VaR, etc.)
- Risk-adjusted metrics (Sharpe, Sortino, Calmar ratios)
- Drawdown analysis
- Regime and factor analysis

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: numpy, pandas, matplotlib, scipy, statsmodels, scikit-learn

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/claudecta.git
cd claudecta

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start Example

```python
from claudecta.backtesting import Backtest
from claudecta.strategies.turtle import TurtleSystem1
from claudecta.risk_management import ATRPositionSizing

# Initialize strategy with position sizing
strategy = TurtleSystem1(
    entry_lookback=20,
    exit_lookback=10,
    position_sizing=ATRPositionSizing(risk_per_trade=0.01)
)

# Run backtest
backtest = Backtest(
    strategy=strategy,
    data_path='data/historical/futures_data.csv',
    initial_capital=100000,
    start_date='2010-01-01',
    end_date='2020-12-31'
)

# Execute and analyze results
results = backtest.run()
backtest.plot_equity_curve()
backtest.print_performance_metrics()
```

## Performance Characteristics

### Classic Turtle System

- Strong performance in trending markets
- Significant drawdowns during choppy markets
- Average win rate: 40-50%
- Win/loss ratio: ≈3:1 (profits 3x larger than losses)
- Typical return during favorable periods: 20-100% annually
- Maximum drawdown: 20-30% historically

### Modern Momentum Strategies

- More consistent performance across market regimes
- Improved Sharpe ratio (typically 0.8-1.5 vs. 0.5-0.8 for classic systems)
- Reduced maximum drawdown (typically 10-20%)
- Better performance during market reversals
- Adaptability to different asset classes and timeframes

## Market Insights

Momentum as a factor has demonstrated remarkable persistence across multiple asset classes and time periods. Research indicates that momentum anomalies stem from a combination of:

1. **Behavioral factors**:
   - Investor under-reaction to new information
   - Herding behavior and trend-following
   - Anchoring bias in price estimates

2. **Institutional factors**:
   - Slow diffusion of information
   - Hedging pressure in futures markets
   - Portfolio rebalancing effects

The effectiveness of momentum strategies varies across market regimes:
- Strongest during clear directional trends
- Challenged during abrupt reversals and regime shifts
- Vulnerable to "momentum crashes" during market rebounds without proper risk management

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Richard Dennis and William Eckhardt for the original Turtle Trading system
- The academic pioneers of momentum research, including Jegadeesh & Titman, Moskowitz & Grinblatt, and Asness et al.
- The quantitative finance community for advancing systematic trading approaches