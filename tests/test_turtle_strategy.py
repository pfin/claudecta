"""
Unit tests for Turtle Trading System strategies.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import strategies to test
from src.strategies.turtle_strategy import TurtleSystem1, TurtleSystem2, DualTurtleSystem


class TestTurtleSystem1(unittest.TestCase):
    """
    Test cases for the Turtle System 1 strategy.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a sample price dataset
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
        
        # Initialize the strategy
        self.strategy = TurtleSystem1(
            entry_lookback=20,
            exit_lookback=10,
            atr_periods=10,
            risk_per_trade=0.01,
            max_units=4
        )
    
    def test_initialization(self):
        """
        Test strategy initialization.
        """
        self.assertEqual(self.strategy.entry_lookback, 20)
        self.assertEqual(self.strategy.exit_lookback, 10)
        self.assertEqual(self.strategy.atr_periods, 10)
        self.assertEqual(self.strategy.risk_per_trade, 0.01)
        self.assertEqual(self.strategy.max_units, 4)
        self.assertTrue(self.strategy.pyramiding)
    
    def test_calculate_atr(self):
        """
        Test ATR calculation.
        """
        high = self.data['TEST_High']
        low = self.data['TEST_Low']
        close = self.data['TEST_Close']
        
        atr = self.strategy.calculate_atr(high, low, close)
        
        # ATR should be a pandas Series
        self.assertIsInstance(atr, pd.Series)
        
        # Length should match input data
        self.assertEqual(len(atr), len(high))
        
        # ATR should be positive
        self.assertTrue((atr >= 0).all())
    
    def test_breakout_levels(self):
        """
        Test breakout level calculation.
        """
        high = self.data['TEST_High']
        low = self.data['TEST_Low']
        
        entry_high, entry_low, exit_high, exit_low = self.strategy.get_breakout_levels(
            high, low, self.strategy.entry_lookback, self.strategy.exit_lookback
        )
        
        # Each level should be a pandas Series
        self.assertIsInstance(entry_high, pd.Series)
        self.assertIsInstance(entry_low, pd.Series)
        self.assertIsInstance(exit_high, pd.Series)
        self.assertIsInstance(exit_low, pd.Series)
        
        # First N values should be NaN where N is the lookback period
        self.assertTrue(pd.isna(entry_high.iloc[self.strategy.entry_lookback-1]))
        self.assertTrue(pd.isna(exit_high.iloc[self.strategy.exit_lookback-1]))
        
        # After warmup, values should be populated
        self.assertFalse(pd.isna(entry_high.iloc[self.strategy.entry_lookback]))
        self.assertFalse(pd.isna(exit_high.iloc[self.strategy.exit_lookback]))
        
        # Entry high should be >= exit high (longer lookback)
        valid_idx = max(self.strategy.entry_lookback, self.strategy.exit_lookback)
        self.assertTrue((entry_high.iloc[valid_idx:] >= exit_high.iloc[valid_idx:]).all())
    
    def test_calculate_unit_size(self):
        """
        Test position sizing calculation.
        """
        equity = 100000
        atr = 2.0
        
        unit_size = self.strategy.calculate_unit_size(equity, atr)
        
        # Unit size should be an integer
        self.assertIsInstance(unit_size, int)
        
        # Unit size should be positive
        self.assertGreater(unit_size, 0)
        
        # Check calculation - risk per trade = 1% of equity = $1000
        # Stop loss = 2 * ATR = 2 * 2 = $4 per unit
        # Unit size = $1000 / $4 = 250 units
        expected_size = int((equity * self.strategy.risk_per_trade) / (2 * atr))
        self.assertEqual(unit_size, expected_size)
    
    def test_calculate_stop_price(self):
        """
        Test stop loss calculation.
        """
        entry_price = 100
        atr = 2.0
        
        # Long stop
        long_stop = self.strategy.calculate_stop_price(entry_price, atr, True)
        expected_long_stop = entry_price - (2 * atr)
        self.assertEqual(long_stop, expected_long_stop)
        
        # Short stop
        short_stop = self.strategy.calculate_stop_price(entry_price, atr, False)
        expected_short_stop = entry_price + (2 * atr)
        self.assertEqual(short_stop, expected_short_stop)
    
    def test_generate_signals_trending_up(self):
        """
        Test signal generation in an uptrend.
        """
        # Create specific uptrend test data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        # First 50 days flat, then uptrend
        prices = [100] * 50 + np.linspace(100, 150, 50).tolist()
        
        # Create OHLC data
        data = {
            'TEST_Open': prices,
            'TEST_High': [p * 1.02 for p in prices],
            'TEST_Low': [p * 0.98 for p in prices],
            'TEST_Close': prices
        }
        
        uptrend_data = pd.DataFrame(data, index=dates)
        
        # Get breakout levels
        high = uptrend_data['TEST_High']
        low = uptrend_data['TEST_Low']
        
        entry_high, entry_low, exit_high, exit_low = self.strategy.get_breakout_levels(
            high, low, self.strategy.entry_lookback, self.strategy.exit_lookback
        )
        
        # At day 60 (after the trend starts), there should be a breakout
        breakout_idx = 60
        
        # We need enough data for valid breakout levels
        breakout_values = {
            'entry_high': entry_high.iloc[breakout_idx-1],
            'entry_low': entry_low.iloc[breakout_idx-1],
            'exit_high': exit_high.iloc[breakout_idx-1],
            'exit_low': exit_low.iloc[breakout_idx-1]
        }
        
        # Mock position and ATR
        position = 0
        units = 0
        atr = 2.0
        equity = 100000
        
        # Generate signal
        signal = self.strategy.generate_signals(
            uptrend_data.iloc[breakout_idx:breakout_idx+1], 
            position, 
            units, 
            breakout_values, 
            atr, 
            equity,
            uptrend_data['TEST_Close'].iloc[breakout_idx]
        )
        
        # Should generate a long signal
        self.assertEqual(signal['direction'], 1)
        self.assertGreater(signal['quantity'], 0)
        self.assertEqual(signal['type'], 'entry_long')


class TestTurtleSystem2(unittest.TestCase):
    """
    Test cases for the Turtle System 2 strategy.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a sample price dataset
        dates = pd.date_range(start='2020-01-01', periods=150, freq='D')
        
        # Create trending price series (up then down)
        prices = np.linspace(100, 150, 75).tolist() + np.linspace(150, 100, 75).tolist()
        
        # Create OHLC data
        data = {
            'TEST_Open': prices,
            'TEST_High': [p * 1.02 for p in prices],
            'TEST_Low': [p * 0.98 for p in prices],
            'TEST_Close': prices
        }
        
        self.data = pd.DataFrame(data, index=dates)
        
        # Initialize the strategy
        self.strategy = TurtleSystem2(
            entry_lookback=55,
            exit_lookback=20,
            atr_periods=20,
            risk_per_trade=0.01,
            max_units=4
        )
    
    def test_initialization(self):
        """
        Test strategy initialization.
        """
        self.assertEqual(self.strategy.entry_lookback, 55)
        self.assertEqual(self.strategy.exit_lookback, 20)
        self.assertEqual(self.strategy.atr_periods, 20)
        self.assertEqual(self.strategy.risk_per_trade, 0.01)
        self.assertEqual(self.strategy.max_units, 4)
        self.assertTrue(self.strategy.pyramiding)
    
    def test_breakout_levels(self):
        """
        Test breakout level calculation.
        """
        high = self.data['TEST_High']
        low = self.data['TEST_Low']
        
        entry_high, entry_low, exit_high, exit_low = self.strategy.get_breakout_levels(
            high, low, self.strategy.entry_lookback, self.strategy.exit_lookback
        )
        
        # Each level should be a pandas Series
        self.assertIsInstance(entry_high, pd.Series)
        self.assertIsInstance(entry_low, pd.Series)
        self.assertIsInstance(exit_high, pd.Series)
        self.assertIsInstance(exit_low, pd.Series)
        
        # First N values should be NaN where N is the lookback period
        self.assertTrue(pd.isna(entry_high.iloc[self.strategy.entry_lookback-1]))
        self.assertTrue(pd.isna(exit_high.iloc[self.strategy.exit_lookback-1]))
        
        # After warmup, values should be populated
        self.assertFalse(pd.isna(entry_high.iloc[self.strategy.entry_lookback]))
        self.assertFalse(pd.isna(exit_high.iloc[self.strategy.exit_lookback]))
        
        # Entry high should be >= exit high (longer lookback)
        valid_idx = max(self.strategy.entry_lookback, self.strategy.exit_lookback)
        self.assertTrue((entry_high.iloc[valid_idx:] >= exit_high.iloc[valid_idx:]).all())


class TestDualTurtleSystem(unittest.TestCase):
    """
    Test cases for the combined Dual Turtle System.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Initialize the strategy
        self.strategy = DualTurtleSystem(
            system1_allocation=0.2,
            system2_allocation=0.8
        )
    
    def test_initialization(self):
        """
        Test strategy initialization.
        """
        self.assertEqual(self.strategy.system1_allocation, 0.2)
        self.assertEqual(self.strategy.system2_allocation, 0.8)
        self.assertIsInstance(self.strategy.system1, TurtleSystem1)
        self.assertIsInstance(self.strategy.system2, TurtleSystem2)
    
    def test_signal_generation(self):
        """
        Test that the dual system generates signals from both components.
        """
        # Create test data
        dates = pd.date_range(start='2020-01-01', periods=150, freq='D')
        prices = np.linspace(100, 150, 75).tolist() + np.linspace(150, 100, 75).tolist()
        
        data = pd.DataFrame({
            'TEST_Open': prices,
            'TEST_High': [p * 1.02 for p in prices],
            'TEST_Low': [p * 0.98 for p in prices],
            'TEST_Close': prices
        }, index=dates)
        
        # Mock inputs for signal generation
        position = 1000  # Current position
        units = {'system1': 1, 'system2': 2}  # Units by system
        
        # Create breakout values
        breakout_values = {
            'system1': {
                'entry_high': 110,
                'entry_low': 90,
                'exit_high': 108,
                'exit_low': 92
            },
            'system2': {
                'entry_high': 115,
                'entry_low': 85,
                'exit_high': 110,
                'exit_low': 90
            }
        }
        
        atr = 2.0
        equity = 100000
        current_price = 105
        
        # Generate signals
        signals = self.strategy.generate_signals(
            data.iloc[100:101],  # Just one row of data
            position,
            units,
            breakout_values,
            atr,
            equity,
            current_price
        )
        
        # Should return signals for both systems
        self.assertIn('system1', signals)
        self.assertIn('system2', signals)


if __name__ == '__main__':
    unittest.main()