"""
Unit tests for the backtesting engine.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import backtesting components to test
from src.backtesting.backtest import Backtest


class MockStrategy:
    """
    Mock strategy for testing the backtesting engine.
    """
    
    def __init__(self, signal_type='constant'):
        self.signal_type = signal_type
        self.initialized = False
    
    def initialize(self, data):
        self.initialized = True
    
    def generate_signals(self, data, current_positions, index, equity):
        if self.signal_type == 'constant':
            # Always generate the same signal
            return {
                'TEST': {
                    'direction': 1,
                    'quantity': 10,
                    'type': 'test_signal'
                }
            }
        elif self.signal_type == 'trend':
            # Generate signals based on trend
            close_col = 'TEST_Close'
            if index > 0:
                if data[close_col].iloc[index] > data[close_col].iloc[index-1]:
                    # Price went up, go long
                    return {
                        'TEST': {
                            'direction': 1,
                            'quantity': 10,
                            'type': 'trend_long'
                        }
                    }
                else:
                    # Price went down, go short
                    return {
                        'TEST': {
                            'direction': -1,
                            'quantity': 10,
                            'type': 'trend_short'
                        }
                    }
            return {}
        elif self.signal_type == 'none':
            # Generate no signals
            return {}
        else:
            return {}


class MockRiskManager:
    """
    Mock risk manager for testing.
    """
    
    def adjust_signals(self, signals, data, index, equity, positions):
        # Reduce all signal quantities by half
        for symbol, signal in signals.items():
            signal['quantity'] = signal['quantity'] // 2
        return signals


class TestBacktest(unittest.TestCase):
    """
    Test cases for the backtesting engine.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create sample price data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        # Create trending price series (up then down)
        prices = np.linspace(100, 150, 50).tolist() + np.linspace(150, 100, 50).tolist()
        
        # Create OHLC data
        data = {
            'TEST_Open': prices,
            'TEST_High': [p * 1.02 for p in prices],
            'TEST_Low': [p * 0.98 for p in prices],
            'TEST_Close': prices
        }
        
        self.data = pd.DataFrame(data, index=dates)
    
    def test_initialization(self):
        """
        Test backtest initialization.
        """
        strategy = MockStrategy()
        
        # Initialize with DataFrame
        backtest = Backtest(
            strategy=strategy,
            data=self.data,
            initial_capital=100000.0,
            commission=0.001,
            slippage=0.0005
        )
        
        self.assertEqual(backtest.initial_capital, 100000.0)
        self.assertEqual(backtest.current_capital, 100000.0)
        self.assertEqual(backtest.commission, 0.001)
        self.assertEqual(backtest.slippage, 0.0005)
        self.assertIs(backtest.strategy, strategy)
        self.assertIs(backtest.data, self.data)
        
        # Initialize with data path
        with self.assertRaises(ValueError):
            # Should raise error if neither data nor data_path is provided
            Backtest(strategy=strategy)
    
    def test_run_with_constant_signals(self):
        """
        Test running backtest with constant signals.
        """
        strategy = MockStrategy(signal_type='constant')
        
        backtest = Backtest(
            strategy=strategy,
            data=self.data,
            initial_capital=100000.0
        )
        
        # Run backtest
        results = backtest.run()
        
        # Check results structure
        self.assertIn('equity_curve', results)
        self.assertIn('trades', results)
        self.assertIn('performance', results)
        
        # Should have generated trades
        self.assertTrue(len(results['trades']) > 0)
        
        # Should have equity curve data
        self.assertEqual(len(results['equity_curve']), len(self.data) - 252)  # Minus warmup
        
        # Should have performance metrics
        self.assertIn('total_return', results['performance'])
        self.assertIn('annualized_return', results['performance'])
        self.assertIn('sharpe_ratio', results['performance'])
    
    def test_run_with_trend_signals(self):
        """
        Test running backtest with trend-based signals.
        """
        strategy = MockStrategy(signal_type='trend')
        
        backtest = Backtest(
            strategy=strategy,
            data=self.data,
            initial_capital=100000.0
        )
        
        # Run backtest
        results = backtest.run()
        
        # Check results
        self.assertTrue(len(results['trades']) > 0)
        
        # Check trade types
        trade_types = [trade['signal_type'] for trade in results['trades']]
        self.assertIn('trend_long', trade_types)
        self.assertIn('trend_short', trade_types)
    
    def test_run_with_no_signals(self):
        """
        Test running backtest with no signals.
        """
        strategy = MockStrategy(signal_type='none')
        
        backtest = Backtest(
            strategy=strategy,
            data=self.data,
            initial_capital=100000.0
        )
        
        # Run backtest
        results = backtest.run()
        
        # Should have no trades
        self.assertEqual(len(results['trades']), 0)
        
        # Equity should remain constant
        equity_values = [point['equity'] for point in results['equity_curve']]
        self.assertTrue(all(eq == 100000.0 for eq in equity_values))
    
    def test_run_with_risk_manager(self):
        """
        Test running backtest with risk management.
        """
        strategy = MockStrategy(signal_type='constant')
        risk_manager = MockRiskManager()
        
        backtest = Backtest(
            strategy=strategy,
            data=self.data,
            initial_capital=100000.0,
            risk_manager=risk_manager
        )
        
        # Run backtest
        results = backtest.run()
        
        # Check first trade quantity (should be halved)
        self.assertEqual(results['trades'][0]['quantity'], 5)
    
    def test_performance_calculation(self):
        """
        Test performance metrics calculation.
        """
        strategy = MockStrategy(signal_type='trend')
        
        backtest = Backtest(
            strategy=strategy,
            data=self.data,
            initial_capital=100000.0
        )
        
        # Run backtest
        results = backtest.run()
        
        # Get performance metrics
        metrics = backtest.get_performance_metrics()
        
        # Check all required metrics are present
        required_metrics = [
            'initial_equity', 'final_equity', 'total_return',
            'annualized_return', 'annualized_volatility', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'calmar_ratio',
            'number_of_trades', 'win_rate'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            
        # Check some basic validations
        self.assertEqual(metrics['initial_equity'], 100000.0)
        self.assertTrue(metrics['final_equity'] != 0)  # Should have some value
        self.assertTrue(isinstance(metrics['total_return'], float))
        self.assertTrue(isinstance(metrics['sharpe_ratio'], float))
    
    def test_calculate_portfolio_value(self):
        """
        Test portfolio value calculation.
        """
        strategy = MockStrategy()
        
        backtest = Backtest(
            strategy=strategy,
            data=self.data,
            initial_capital=100000.0
        )
        
        # Set up positions
        backtest.positions = {
            'TEST': {'quantity': 100, 'avg_price': 120.0}
        }
        
        # Calculate portfolio value
        portfolio_value = backtest._calculate_portfolio_value(self.data.iloc[50])
        
        # Expected value: 100 shares * price at index 50
        expected_value = 100 * self.data['TEST_Close'].iloc[50]
        self.assertEqual(portfolio_value, expected_value)
    
    def test_get_price_column(self):
        """
        Test getting price column name.
        """
        strategy = MockStrategy()
        
        backtest = Backtest(
            strategy=strategy,
            data=self.data,
            initial_capital=100000.0
        )
        
        # Get column for TEST
        col = backtest._get_price_column('TEST')
        self.assertEqual(col, 'TEST_Close')
        
        # Test with non-existent symbol
        with self.assertRaises(ValueError):
            backtest._get_price_column('NONEXISTENT')


if __name__ == '__main__':
    unittest.main()