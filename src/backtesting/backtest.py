"""
Core backtesting engine for momentum trading strategies.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import matplotlib.pyplot as plt
from datetime import datetime


class Backtest:
    """
    A backtesting engine for trading strategies.
    
    Handles data processing, strategy execution, portfolio management, 
    and performance reporting for backtesting trading strategies.
    """
    
    def __init__(
        self,
        strategy: Any,
        data: Optional[pd.DataFrame] = None,
        data_path: Optional[str] = None,
        initial_capital: float = 100000.0,
        commission: float = 0.0,
        slippage: float = 0.0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        risk_manager: Optional[Any] = None
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            strategy: The trading strategy to backtest
            data: Pandas DataFrame with historical price data (optional if data_path provided)
            data_path: Path to CSV file with historical data (optional if data provided)
            initial_capital: Starting capital for the backtest
            commission: Commission rate per trade (percentage as decimal)
            slippage: Slippage per trade (percentage as decimal)
            start_date: Start date for the backtest (YYYY-MM-DD)
            end_date: End date for the backtest (YYYY-MM-DD)
            risk_manager: Optional risk management component
        """
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.start_date = start_date
        self.end_date = end_date
        
        # Load and prepare data
        if data is not None:
            self.data = data
        elif data_path is not None:
            self.data = self._load_data(data_path)
        else:
            raise ValueError("Either data or data_path must be provided")
        
        # Portfolio and results tracking
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load historical data from a CSV file.
        
        Args:
            data_path: Path to the CSV file
            
        Returns:
            DataFrame with historical data
        """
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Filter by date range if provided
        if self.start_date:
            data = data[data.index >= self.start_date]
        if self.end_date:
            data = data[data.index <= self.end_date]
            
        return data
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the backtest.
        
        Returns:
            Dictionary with backtest results
        """
        # Reset tracking variables
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # Initialize the strategy
        if hasattr(self.strategy, 'initialize'):
            self.strategy.initialize(self.data)
        
        # Iterate through each data point
        for i, (timestamp, row) in enumerate(self.data.iterrows()):
            # Skip warm-up period if needed
            if i < 252 and len(self.data) > 252:  # Default warm-up of 1 year
                continue
                
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(row)
            
            # Record equity at this point
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self.current_capital + portfolio_value,
                'cash': self.current_capital
            })
            
            # Generate trading signals
            current_positions = {symbol: position['quantity'] for symbol, position in self.positions.items()}
            
            signals = self.strategy.generate_signals(
                self.data, 
                current_positions, 
                i, 
                self.current_capital + portfolio_value
            )
            
            # Apply risk management if provided
            if self.risk_manager:
                signals = self._apply_risk_management(signals, i, self.current_capital + portfolio_value)
            
            # Execute trades based on signals
            for symbol, signal in signals.items():
                self._execute_trade(timestamp, symbol, signal, row)
        
        # Calculate final performance metrics
        self._calculate_performance_metrics()
        
        return {
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'performance': self.performance_metrics
        }
    
    def _apply_risk_management(
        self, 
        signals: Dict[str, Dict[str, Any]], 
        index: int, 
        equity: float
    ) -> Dict[str, Dict[str, Any]]:
        """
        Apply risk management to signals.
        
        Args:
            signals: Trading signals
            index: Current index in the data
            equity: Current equity
            
        Returns:
            Adjusted signals
        """
        # Risk management depends on the specific implementation
        # This is just a placeholder for custom risk management logic
        if hasattr(self.risk_manager, 'adjust_signals'):
            return self.risk_manager.adjust_signals(signals, self.data, index, equity, self.positions)
        
        return signals
    
    def _calculate_portfolio_value(self, current_prices: pd.Series) -> float:
        """
        Calculate the current portfolio value based on positions and prices.
        
        Args:
            current_prices: Current prices for all assets
            
        Returns:
            Total portfolio value
        """
        portfolio_value = 0.0
        
        for symbol, position in self.positions.items():
            price_col = self._get_price_column(symbol)
            if price_col in current_prices:
                portfolio_value += position['quantity'] * current_prices[price_col]
                
        return portfolio_value
    
    def _get_price_column(self, symbol: str) -> str:
        """
        Get the price column name for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Column name for price data
        """
        # Check different possible column formats
        if f"{symbol}_Close" in self.data.columns:
            return f"{symbol}_Close"
        elif (symbol, 'Close') in self.data.columns:
            return (symbol, 'Close')
        else:
            # Try to find any column with the symbol name
            for col in self.data.columns:
                if isinstance(col, str) and col.startswith(f"{symbol}_"):
                    return col
                elif isinstance(col, tuple) and col[0] == symbol:
                    return col
                    
            raise ValueError(f"Price data not found for symbol: {symbol}")
    
    def _execute_trade(
        self, 
        timestamp: pd.Timestamp, 
        symbol: str, 
        signal: Dict[str, Any],
        prices: pd.Series
    ) -> None:
        """
        Execute a trade based on a signal.
        
        Args:
            timestamp: Current timestamp
            symbol: Symbol to trade
            signal: Trading signal details
            prices: Current price data
        """
        price_col = self._get_price_column(symbol)
        if price_col not in prices:
            return
            
        current_price = prices[price_col]
        quantity = signal.get('quantity', 0)
        direction = signal.get('direction', 0)  # 1 for buy, -1 for sell
        
        # Skip if no action
        if direction == 0 or quantity == 0:
            return
            
        # Calculate actual execution price with slippage
        slippage_factor = 1 + (self.slippage * direction)
        execution_price = current_price * slippage_factor
        
        # Calculate commission
        trade_value = execution_price * quantity
        commission_amount = trade_value * self.commission
        
        # Update positions
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0, 'avg_price': 0}
            
        # Buy
        if direction > 0:
            # Check if enough capital
            total_cost = trade_value + commission_amount
            if total_cost > self.current_capital:
                # Adjust quantity if not enough capital
                quantity = int((self.current_capital - commission_amount) / execution_price)
                if quantity <= 0:
                    return
                    
                trade_value = execution_price * quantity
                commission_amount = trade_value * self.commission
                
            # Update position
            old_quantity = self.positions[symbol]['quantity']
            old_avg_price = self.positions[symbol]['avg_price']
            
            new_quantity = old_quantity + quantity
            new_avg_price = ((old_quantity * old_avg_price) + (quantity * execution_price)) / new_quantity
            
            self.positions[symbol]['quantity'] = new_quantity
            self.positions[symbol]['avg_price'] = new_avg_price
            
            # Update capital
            self.current_capital -= (trade_value + commission_amount)
            
        # Sell
        elif direction < 0:
            # Check if enough position
            current_quantity = self.positions[symbol]['quantity']
            if abs(quantity) > current_quantity:
                quantity = -current_quantity
                
            if quantity == 0:
                return
                
            trade_value = execution_price * abs(quantity)
            commission_amount = trade_value * self.commission
            
            # Update position
            self.positions[symbol]['quantity'] += quantity  # quantity is negative for sells
            
            # Remove position if quantity is zero
            if self.positions[symbol]['quantity'] == 0:
                avg_price = self.positions[symbol]['avg_price']
                del self.positions[symbol]
            
            # Update capital
            self.current_capital += (trade_value - commission_amount)
        
        # Record trade
        self.trades.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'price': execution_price,
            'commission': commission_amount,
            'value': trade_value,
            'signal_type': signal.get('type', 'unknown')
        })
    
    def _calculate_performance_metrics(self) -> None:
        """
        Calculate performance metrics from the backtest results.
        """
        if not self.equity_curve:
            self.performance_metrics = {}
            return
            
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Initial and final equity
        initial_equity = self.initial_capital
        final_equity = equity_df['equity'].iloc[-1]
        
        # Total return
        total_return = (final_equity / initial_equity) - 1
        
        # Annualized return
        days = (equity_df.index[-1] - equity_df.index[0]).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # Volatility (annualized)
        daily_volatility = equity_df['returns'].std()
        annualized_volatility = daily_volatility * (252 ** 0.5) if not pd.isnull(daily_volatility) else 0
        
        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = equity_df['returns'][equity_df['returns'] < 0]
        downside_volatility = downside_returns.std() * (252 ** 0.5) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
        
        # Maximum drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] / equity_df['cummax']) - 1
        max_drawdown = equity_df['drawdown'].min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Number of trades
        num_trades = len(self.trades)
        
        # Win rate
        if num_trades > 0:
            trades_df = pd.DataFrame(self.trades)
            trades_df['profit'] = 0
            
            # Calculate profit/loss for each trade
            for i, trade in trades_df.iterrows():
                if trade['direction'] > 0:  # Buy
                    for j in range(i+1, len(trades_df)):
                        future_trade = trades_df.iloc[j]
                        if future_trade['symbol'] == trade['symbol'] and future_trade['direction'] < 0:
                            trades_df.at[i, 'profit'] += (future_trade['price'] - trade['price']) * min(trade['quantity'], abs(future_trade['quantity']))
                            break
                            
            winning_trades = len(trades_df[trades_df['profit'] > 0])
            win_rate = winning_trades / num_trades
        else:
            win_rate = 0
        
        self.performance_metrics = {
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'number_of_trades': num_trades,
            'win_rate': win_rate
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get the performance metrics from the backtest.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.performance_metrics:
            self._calculate_performance_metrics()
            
        return self.performance_metrics
    
    def print_performance_metrics(self) -> None:
        """
        Print formatted performance metrics.
        """
        metrics = self.get_performance_metrics()
        
        print("\nPerformance Metrics:")
        print(f"Initial Equity: ${metrics['initial_equity']:,.2f}")
        print(f"Final Equity: ${metrics['final_equity']:,.2f}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
        print(f"Number of Trades: {metrics['number_of_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
    
    def plot_equity_curve(self) -> None:
        """
        Plot the equity curve from the backtest.
        """
        try:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            
            plt.figure(figsize=(12, 6))
            plt.plot(equity_df.index, equity_df['equity'])
            plt.title('Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Equity')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib is required for plotting")
    
    def plot_drawdowns(self) -> None:
        """
        Plot the drawdowns from the backtest.
        """
        try:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            equity_df['cummax'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] / equity_df['cummax']) - 1
            
            plt.figure(figsize=(12, 6))
            plt.plot(equity_df.index, equity_df['drawdown'] * 100)
            plt.title('Drawdowns')
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib is required for plotting")
    
    def plot_performance_dashboard(self) -> None:
        """
        Plot a comprehensive performance dashboard.
        """
        try:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            equity_df['returns'] = equity_df['equity'].pct_change()
            equity_df['cumulative'] = (1 + equity_df['returns']).cumprod()
            equity_df['cummax'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] / equity_df['cummax']) - 1
            
            # Calculate rolling metrics
            window = min(252, len(equity_df) // 2)  # Use 1 year or half the data
            if window > 0:
                equity_df['rolling_return'] = equity_df['returns'].rolling(window).mean() * 252
                equity_df['rolling_vol'] = equity_df['returns'].rolling(window).std() * (252 ** 0.5)
                equity_df['rolling_sharpe'] = equity_df['rolling_return'] / equity_df['rolling_vol']
            
            # Create dashboard
            fig, axes = plt.subplots(4, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [3, 2, 2, 2]})
            
            # Equity curve
            axes[0].plot(equity_df.index, equity_df['equity'])
            axes[0].set_title('Equity Curve')
            axes[0].set_ylabel('Equity ($)')
            axes[0].grid(True)
            
            # Drawdowns
            axes[1].fill_between(equity_df.index, equity_df['drawdown'] * 100, 0, alpha=0.3, color='red')
            axes[1].plot(equity_df.index, equity_df['drawdown'] * 100, color='red')
            axes[1].set_title('Drawdowns')
            axes[1].set_ylabel('Drawdown (%)')
            axes[1].grid(True)
            
            # Rolling metrics
            if window > 0 and len(equity_df) > window:
                ax2 = axes[2].twinx()
                axes[2].plot(equity_df.index, equity_df['rolling_return'] * 100, 'g-', label='Return')
                ax2.plot(equity_df.index, equity_df['rolling_vol'] * 100, 'b--', label='Volatility')
                axes[2].set_title('Rolling 1-Year Return and Volatility')
                axes[2].set_ylabel('Return (%)', color='g')
                ax2.set_ylabel('Volatility (%)', color='b')
                axes[2].grid(True)
                
                # Rolling Sharpe
                axes[3].plot(equity_df.index, equity_df['rolling_sharpe'], 'purple')
                axes[3].set_title('Rolling 1-Year Sharpe Ratio')
                axes[3].set_ylabel('Sharpe Ratio')
                axes[3].grid(True)
            
            plt.tight_layout()
            plt.show()
            
            # Trade analysis
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                
                # Plot trade distribution
                plt.figure(figsize=(12, 6))
                
                # Group by month
                trades_df['month'] = trades_df['timestamp'].dt.to_period('M')
                monthly_trades = trades_df.groupby('month').size()
                
                plt.bar(range(len(monthly_trades)), monthly_trades.values)
                plt.xticks(range(len(monthly_trades)), [str(m) for m in monthly_trades.index], rotation=90)
                plt.title('Monthly Trade Distribution')
                plt.xlabel('Month')
                plt.ylabel('Number of Trades')
                plt.tight_layout()
                plt.show()
        except ImportError:
            print("Matplotlib is required for plotting")
    
    def run_monte_carlo_analysis(
        self, 
        iterations: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo analysis on the backtest results.
        
        Args:
            iterations: Number of Monte Carlo iterations
            confidence_level: Confidence level for metrics
            
        Returns:
            Dictionary with Monte Carlo results
        """
        try:
            # Get returns series
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            returns = equity_df['equity'].pct_change().dropna()
            
            # Run Monte Carlo simulation
            sim_results = []
            
            for _ in range(iterations):
                # Resample returns with replacement
                sim_returns = returns.sample(n=len(returns), replace=True)
                
                # Calculate cumulative performance
                sim_equity = self.initial_capital * (1 + sim_returns).cumprod()
                
                # Calculate metrics for this iteration
                final_equity = sim_equity.iloc[-1]
                total_return = (final_equity / self.initial_capital) - 1
                max_drawdown = (sim_equity / sim_equity.cummax() - 1).min()
                
                sim_results.append({
                    'final_equity': final_equity,
                    'total_return': total_return,
                    'max_drawdown': max_drawdown
                })
                
            # Convert results to DataFrame
            sim_df = pd.DataFrame(sim_results)
            
            # Calculate statistics
            lower_idx = int(iterations * (1 - confidence_level) / 2)
            upper_idx = int(iterations * (1 + confidence_level) / 2)
            
            sorted_returns = sorted(sim_df['total_return'])
            sorted_drawdowns = sorted(sim_df['max_drawdown'])
            
            # Calculate confidence intervals
            return_ci = (sorted_returns[lower_idx], sorted_returns[upper_idx])
            drawdown_ci = (sorted_drawdowns[lower_idx], sorted_drawdowns[upper_idx])
            
            # Plot distribution of returns
            plt.figure(figsize=(12, 6))
            plt.hist(sim_df['total_return'] * 100, bins=50, alpha=0.5)
            plt.axvline(return_ci[0] * 100, color='r', linestyle='--', 
                        label=f'{confidence_level*100:.0f}% CI Lower: {return_ci[0]:.2%}')
            plt.axvline(return_ci[1] * 100, color='r', linestyle='--',
                        label=f'{confidence_level*100:.0f}% CI Upper: {return_ci[1]:.2%}')
            plt.title('Monte Carlo: Distribution of Returns')
            plt.xlabel('Total Return (%)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            # Plot distribution of drawdowns
            plt.figure(figsize=(12, 6))
            plt.hist(sim_df['max_drawdown'] * 100, bins=50, alpha=0.5)
            plt.axvline(drawdown_ci[0] * 100, color='r', linestyle='--',
                        label=f'{confidence_level*100:.0f}% CI Lower: {drawdown_ci[0]:.2%}')
            plt.axvline(drawdown_ci[1] * 100, color='r', linestyle='--',
                        label=f'{confidence_level*100:.0f}% CI Upper: {drawdown_ci[1]:.2%}')
            plt.title('Monte Carlo: Distribution of Maximum Drawdowns')
            plt.xlabel('Maximum Drawdown (%)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            # Return results
            monte_carlo_results = {
                'iterations': iterations,
                'confidence_level': confidence_level,
                'return_mean': sim_df['total_return'].mean(),
                'return_std': sim_df['total_return'].std(),
                'return_ci': return_ci,
                'drawdown_mean': sim_df['max_drawdown'].mean(),
                'drawdown_std': sim_df['max_drawdown'].std(),
                'drawdown_ci': drawdown_ci
            }
            
            return monte_carlo_results
        except ImportError:
            print("Required packages not available for Monte Carlo analysis")
            return {}


if __name__ == "__main__":
    print("Backtesting engine module imported")