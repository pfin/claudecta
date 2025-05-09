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

5. **Advanced Signal Processing**:
   - Multi-frequency momentum decomposition
   - Wavelet analysis for trend identification
   - Signal smoothing and denoising techniques
   - Kalman filtering for parameter adaptation

6. **Covariance Matrix Estimation Techniques**:
   - Shrinkage estimators to improve stability
   - Factor-based covariance modeling
   - Graphical LASSO for sparse precision matrix estimation
   - Time-varying correlation models (DCC-GARCH)

7. **Regularization Approaches**:
   - L1/L2 regularization for signal generation
   - Transaction cost penalties in optimization
   - Turnover constraints for position stability
   - Position concentration limits

8. **Alternative Data Integration**:
   - Sentiment signals from news and social media
   - Order flow and market microstructure data
   - Macroeconomic indicators and central bank policy
   - Satellite imagery and other non-traditional datasets

## Advanced Implementation Details

### Covariance Matrix Estimation

Accurate covariance estimation is critical for portfolio construction in momentum strategies:

```python
def estimate_covariance(returns, method="shrinkage", **kwargs):
    """
    Estimate covariance matrix with various methods.
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        Asset returns data
    method : str
        Method to use ('sample', 'shrinkage', 'factor', 'graphical_lasso')
    
    Returns:
    --------
    pandas.DataFrame
        Estimated covariance matrix
    """
    if method == "sample":
        # Simple sample covariance
        return returns.cov()
        
    elif method == "shrinkage":
        # Ledoit-Wolf shrinkage estimator
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        cov = lw.fit(returns).covariance_
        return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
        
    elif method == "factor":
        # Factor-based covariance (simplified PCA approach)
        from sklearn.decomposition import PCA
        n_factors = kwargs.get("n_factors", 3)
        
        # Extract factors
        pca = PCA(n_components=n_factors)
        factors = pca.fit_transform(returns)
        loadings = pca.components_.T
        
        # Reconstruct covariance
        factor_cov = np.cov(factors, rowvar=False)
        common_cov = loadings @ factor_cov @ loadings.T
        specific_var = np.diag(np.var(returns - factors @ loadings.T, axis=0))
        
        cov = common_cov + specific_var
        return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
        
    elif method == "graphical_lasso":
        # Sparse precision matrix estimation
        from sklearn.covariance import GraphicalLassoCV
        model = GraphicalLassoCV(alphas=4)
        model.fit(returns)
        
        cov = model.covariance_
        return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    
    else:
        raise ValueError(f"Unknown method: {method}")
```

### Regularized Portfolio Optimization

Momentum strategies often benefit from regularized optimization to control turnover and risk:

```python
def optimize_portfolio(expected_returns, covariance, current_weights=None, 
                      target_volatility=0.1, l1_reg=0.01, turnover_penalty=0.005):
    """
    Optimize portfolio with regularization constraints.
    
    Parameters:
    -----------
    expected_returns : pandas.Series
        Expected returns for each asset
    covariance : pandas.DataFrame
        Covariance matrix of returns
    current_weights : pandas.Series
        Current portfolio weights
    target_volatility : float
        Target portfolio volatility
    l1_reg : float
        L1 regularization parameter (for sparsity)
    turnover_penalty : float
        Penalty for turnover
        
    Returns:
    --------
    pandas.Series
        Optimized portfolio weights
    """
    import cvxpy as cp
    
    n = len(expected_returns)
    assets = expected_returns.index
    
    # Define variables and parameters
    w = cp.Variable(n)
    gamma = cp.Parameter(nonneg=True)
    
    # Define risk
    risk = cp.quad_form(w, covariance)
    
    # L1 regularization for sparsity
    reg_l1 = cp.norm1(w)
    
    # Turnover penalty
    turnover = 0
    if current_weights is not None:
        curr_w = np.array(current_weights)
        turnover = cp.norm1(w - curr_w)
    
    # Define objective
    expected_return = expected_returns @ w
    objective = cp.Maximize(expected_return - gamma * risk - l1_reg * reg_l1 - turnover_penalty * turnover)
    
    # Define constraints
    constraints = [
        cp.sum(w) == 1,  # Fully invested
        w >= -0.5,       # Long/short constraint
        w <= 0.5,        # Position limit
        risk <= target_volatility**2  # Volatility constraint
    ]
    
    # Solve the problem
    gamma.value = 1.0  # Risk aversion parameter
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    # Extract solution
    if w.value is None:
        raise ValueError("Optimization failed")
        
    return pd.Series(w.value, index=assets)
```

### Adaptive Timeframe Selection

Modern momentum strategies often adapt their lookback periods based on market conditions:

```python
def adaptive_lookback_period(prices, min_period=20, max_period=252, 
                            adaptation_speed=0.1, method="autocorrelation"):
    """
    Dynamically adapt lookback period based on market conditions.
    
    Parameters:
    -----------
    prices : pandas.Series
        Price series
    min_period : int
        Minimum lookback period
    max_period : int
        Maximum lookback period
    adaptation_speed : float
        How quickly to adapt (0-1)
    method : str
        Method for adaptation ('autocorrelation', 'volatility', 'trend_strength')
        
    Returns:
    --------
    int
        Adapted lookback period
    """
    returns = prices.pct_change().dropna()
    
    if method == "autocorrelation":
        # Use autocorrelation structure
        # Look for strongest positive autocorrelation within range
        acf_values = []
        for lag in range(min_period, max_period + 1, 10):
            if len(returns) > lag + 10:
                # Calculate autocorrelation at this lag
                acf = returns.iloc[:-lag].corr(returns.iloc[lag:])
                acf_values.append((lag, acf))
        
        if not acf_values:
            return min_period
            
        # Sort by strongest positive autocorrelation
        acf_values.sort(key=lambda x: -x[1])
        best_lag = acf_values[0][0]
        
        return int(best_lag)
        
    elif method == "volatility":
        # Adapt based on volatility regime
        # Higher volatility → shorter lookback periods
        recent_vol = returns.iloc[-63:].std() * np.sqrt(252)
        vol_percentile = returns.rolling(252).std().rank(pct=True).iloc[-1]
        
        # Scale between min and max periods based on volatility percentile
        # Lower volatility → longer lookback
        period = min_period + (max_period - min_period) * (1 - vol_percentile)
        return int(period)
        
    elif method == "trend_strength":
        # Adapt based on trend strength
        # Calculate various trend strength indicators
        ma_ratio = prices.iloc[-1] / prices.rolling(63).mean().iloc[-1] - 1
        
        # Scale period based on trend strength
        # Stronger trends → longer lookback
        trend_score = min(max(ma_ratio * 10, 0), 1)  # Scale between 0-1
        period = min_period + (max_period - min_period) * trend_score
        return int(period)
        
    else:
        return min_period
```

## Example Implementation: Time-Series Momentum with Volatility Scaling

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load historical data
def load_data(symbol, start_date, end_date):
    # Example using yfinance - in production you'd use your own data source
    import yfinance as yf
    data = yf.download(symbol, start=start_date, end=end_date)
    return data['Adj Close']

# Calculate time-series momentum signal
def calculate_momentum_signal(prices, lookback=252):
    # Calculate returns over lookback period
    momentum = prices.pct_change(periods=lookback)
    
    # Convert to signal (-1 to 1)
    signal = np.sign(momentum)
    
    # Optional: Scale by strength
    # signal = momentum / momentum.rolling(lookback).std()
    # signal = np.clip(signal, -1, 1)  # Clip to -1 to 1 range
    
    return signal

# Calculate volatility for scaling
def calculate_volatility(prices, lookback=63):
    returns = prices.pct_change().dropna()
    volatility = returns.rolling(window=lookback).std() * np.sqrt(252)
    return volatility

# Generate volatility-scaled positions
def calculate_positions(signals, volatility, equity, prices, target_vol=0.1):
    # Apply inverse volatility scaling
    # Formula: signal * (target_vol / asset_vol) * equity / price
    positions = signals * (target_vol / volatility) * equity / prices
    
    # Convert to integer position sizes
    positions = positions.round().astype(int)
    
    return positions

# Calculate strategy performance
def calculate_performance(positions, returns):
    # Calculate strategy returns
    strategy_returns = positions.shift() * returns
    
    # Calculate cumulative performance
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # Calculate metrics
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
    
    return {
        'returns': strategy_returns,
        'cumulative': cumulative_returns,
        'sharpe': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

# Run backtest
def run_backtest(symbol='SPY', start_date='2000-01-01', end_date='2020-12-31',
                lookback=252, vol_lookback=63, target_vol=0.1):
    # Load data
    prices = load_data(symbol, start_date, end_date)
    returns = prices.pct_change().dropna()
    
    # Calculate signals
    signals = calculate_momentum_signal(prices, lookback)
    volatility = calculate_volatility(prices, vol_lookback)
    
    # Generate positions
    initial_equity = 1000000  # $1M initial capital
    positions = calculate_positions(signals, volatility, initial_equity, prices, target_vol)
    
    # Calculate performance
    performance = calculate_performance(positions, returns)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(prices.index, prices)
    plt.title(f'{symbol} Price')
    
    plt.subplot(3, 1, 2)
    plt.plot(signals.index, signals)
    plt.title('Momentum Signal')
    
    plt.subplot(3, 1, 3)
    plt.plot(performance['cumulative'].index, performance['cumulative'])
    plt.title(f'Strategy Performance (Sharpe: {performance["sharpe"]:.2f}, Max DD: {performance["max_drawdown"]:.2%})')
    
    plt.tight_layout()
    plt.show()
    
    return performance
```

## Project Components

### Strategies

- **Turtle Trading System**
  - Original System 1 (20-day breakout)
  - Original System 2 (55-day breakout)

- **Modern Momentum Strategies**
  - Time-Series Momentum
  - Cross-Sectional Momentum
  - Adaptive Momentum (regime-aware)
  - Dynamic Timeframe Selection
  - Multi-signal Combination Approaches
  - Model-based Prediction Systems

### Risk Management

- **Position Sizing**
  - Fixed Dollar
  - Fixed Risk (percentage)
  - ATR-based (Turtle-style)
  - Volatility-scaled
  - Optimized sizing with constraints

- **Volatility Scaling**
  - Constant Volatility Targeting
  - Dynamic Volatility Targeting
  - Adaptive Volatility Targeting
  - Regime-based Adjustments
  - Conditional Volatility Modeling (GARCH)

- **Portfolio Construction**
  - Naive equal weighting
  - Inverse volatility weighting
  - Mean-variance optimization
  - Risk parity approaches
  - Black-Litterman framework
  - Hierarchical risk parity
  - Nested clustered optimization

- **Drawdown Control**
  - Time-based exposure scaling
  - Conditional Value-at-Risk (CVaR) constraints
  - Dynamic leverage management
  - Tactical hedging during market stress
  - Option overlay protection strategies

### Backtesting Engine

- Event-driven architecture
- Multiple asset class support
- Transaction cost modeling
- Realistic simulation of position sizing and risk management
- Comprehensive performance metrics
- Statistical significance testing
- Monte Carlo simulation
- Walk-forward optimization

### Performance Analysis

- Return metrics (total, annualized, etc.)
- Risk metrics (volatility, drawdown, VaR, etc.)
- Risk-adjusted metrics (Sharpe, Sortino, Calmar ratios)
- Drawdown analysis
- Regime and factor analysis
- Attribution analysis (by asset, sector, factor)
- Performance fingerprinting
- Stress-testing and scenario analysis

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: numpy, pandas, matplotlib, scipy, statsmodels, scikit-learn, cvxpy

### Installation

```bash
# Clone the repository
git clone https://github.com/pfin/claudecta.git
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

### Modern Momentum Example

```python
from claudecta.backtesting import Backtest
from claudecta.strategies.momentum import TimeSeriesMomentum
from claudecta.risk_management.volatility import DynamicVolatilityTargeting

# Initialize modern momentum strategy
strategy = TimeSeriesMomentum(
    lookback_periods=[60, 120, 252],  # Multiple lookback periods
    volatility_lookback=63,
    target_volatility=0.10,
    risk_scaling=True,
    rebalance_frequency=5,
    use_trend_filter=True
)

# Set up risk management
risk_manager = DynamicVolatilityTargeting(
    base_target_vol=0.10,
    max_leverage=2.0,
    vol_regime_detection='adaptive'
)

# Run backtest with transaction costs
backtest = Backtest(
    strategy=strategy,
    risk_manager=risk_manager,
    data_path='data/historical/multi_asset.csv',
    initial_capital=1000000,
    commission=0.0005,  # 5 bps per trade
    slippage=0.0002,    # 2 bps slippage
    start_date='2005-01-01',
    end_date='2020-12-31'
)

# Execute and analyze
results = backtest.run()
backtest.plot_performance_dashboard()
backtest.run_monte_carlo_analysis(iterations=1000)
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
- Typical annual returns of 10-15% with reduced volatility
- Lower correlation to traditional assets
- Improved performance in crisis periods with proper risk management

## Market Insights

Momentum as a factor has demonstrated remarkable persistence across multiple asset classes and time periods. Research indicates that momentum anomalies stem from a combination of:

1. **Behavioral factors**:
   - Investor under-reaction to new information
   - Herding behavior and trend-following
   - Anchoring bias in price estimates
   - Disposition effect (selling winners too early, holding losers too long)
   - Confirmation bias reinforcing existing trends

2. **Institutional factors**:
   - Slow diffusion of information
   - Hedging pressure in futures markets
   - Portfolio rebalancing effects
   - Fund flow dynamics
   - Institutional mandate constraints

3. **Risk-based explanations**:
   - Time-varying risk premia
   - Compensation for crash risk
   - Exposure to systematic factors
   - Liquidity provision during trending markets

The effectiveness of momentum strategies varies across market regimes:
- Strongest during clear directional trends
- Challenged during abrupt reversals and regime shifts
- Vulnerable to "momentum crashes" during market rebounds without proper risk management
- Enhanced performance when combined with complementary factors (value, carry, etc.)

## Real-World Applications

ClaudeCTA can be used for various applications in quantitative finance:

1. **Managed Futures / CTA Strategies**:
   - Systematic trend-following across multiple asset classes
   - Global macro portfolio diversification
   - Crisis alpha generation

2. **Long/Short Equity Strategies**:
   - Factor-based stock selection
   - Sector rotation
   - Statistical arbitrage

3. **Multi-Asset Allocation**:
   - Tactical asset allocation
   - Risk parity implementation
   - Adaptive portfolio management

4. **Research and Education**:
   - Academic research on factor investing
   - Strategy development and testing
   - Educational demonstrations of quantitative concepts

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Richard Dennis and William Eckhardt for the original Turtle Trading system
- The academic pioneers of momentum research, including Jegadeesh & Titman, Moskowitz & Ooi & Pedersen, and Asness et al.
- The quantitative finance community for advancing systematic trading approaches