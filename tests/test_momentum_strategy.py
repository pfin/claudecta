"""
Unit tests for modern momentum strategy implementations.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import strategies to test
from src.strategies.momentum_strategy import TimeSeriesMomentum, CrossSectionalMomentum, HybridMomentumStrategy


class TestTimeSeriesMomentum(unittest.TestCase):
    """
    Test cases for the Time-Series Momentum strategy.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create sample price data for multiple assets
        dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
        
        # Asset A: Uptrend
        prices_a = np.linspace(100, 200, 300)
        
        # Asset B: Downtrend
        prices_b = np.linspace(200, 100, 300)
        
        # Asset C: Sideways
        prices_c = np.sin(np.linspace(0, 5*np.pi, 300)) * 10 + 150
        
        # Create DataFrame
        data = {
            'A_Close': prices_a,
            'B_Close': prices_b,
            'C_Close': prices_c
        }
        
        self.data = pd.DataFrame(data, index=dates)
        
        # Initialize the strategy
        self.strategy = TimeSeriesMomentum(
            lookback_periods=[60, 120],
            volatility_lookback=63,
            target_volatility=0.10,
            risk_scaling=True,
            rebalance_frequency=5,
            max_leverage=2.0
        )
    
    def test_initialization(self):
        """
        Test strategy initialization.
        """
        self.assertEqual(self.strategy.lookback_periods, [60, 120])
        self.assertEqual(self.strategy.volatility_lookback, 63)
        self.assertEqual(self.strategy.target_volatility, 0.10)
        self.assertTrue(self.strategy.risk_scaling)
        self.assertEqual(self.strategy.rebalance_frequency, 5)
        self.assertEqual(self.strategy.max_leverage, 2.0)
    
    def test_asset_name_extraction(self):
        """
        Test asset name extraction from DataFrame.
        """
        asset_names = self.strategy._get_asset_names(self.data)
        
        # Should extract A, B, C
        self.assertIn('A', asset_names)
        self.assertIn('B', asset_names)
        self.assertIn('C', asset_names)
        self.assertEqual(len(asset_names), 3)
    
    def test_asset_price_extraction(self):
        """
        Test asset price extraction.
        """
        # Test for asset A
        prices_a = self.strategy._get_asset_prices(self.data, 'A')
        
        # Should return the A_Close series
        self.assertIsInstance(prices_a, pd.Series)
        self.assertEqual(len(prices_a), len(self.data))
        self.assertEqual(prices_a.iloc[0], self.data['A_Close'].iloc[0])
    
    def test_momentum_signal_calculation(self):
        """
        Test momentum signal calculation.
        """
        # Calculate momentum signals
        signals = self.strategy.calculate_momentum_signals(self.data)
        
        # Should have columns for each asset and lookback period
        for asset in ['A', 'B', 'C']:
            for period in self.strategy.lookback_periods:
                col_name = f"{asset}_momentum_{period}"
                self.assertIn(col_name, signals.columns)
        
        # First N values should be NaN where N is the lookback period
        max_lookback = max(self.strategy.lookback_periods)
        for asset in ['A', 'B', 'C']:
            col_name = f"{asset}_momentum_{max_lookback}"
            self.assertTrue(pd.isna(signals[col_name].iloc[max_lookback-1]))
            self.assertFalse(pd.isna(signals[col_name].iloc[max_lookback]))
            
        # Asset A should have positive momentum (uptrend)
        for period in self.strategy.lookback_periods:
            col_name = f"A_momentum_{period}"
            self.assertTrue((signals[col_name].iloc[max_lookback:] > 0).all())
            
        # Asset B should have negative momentum (downtrend)
        for period in self.strategy.lookback_periods:
            col_name = f"B_momentum_{period}"
            self.assertTrue((signals[col_name].iloc[max_lookback:] < 0).all())
    
    def test_volatility_calculation(self):
        """
        Test volatility calculation.
        """
        # Calculate volatility
        volatility = self.strategy.calculate_volatility(self.data)
        
        # Should have columns for each asset
        for asset in ['A', 'B', 'C']:
            vol_col = f"{asset}_volatility"
            self.assertIn(vol_col, volatility.columns)
            
        # Volatility should be positive
        for asset in ['A', 'B', 'C']:
            vol_col = f"{asset}_volatility"
            self.assertTrue((volatility[vol_col].dropna() > 0).all())
    
    def test_weight_calculation(self):
        """
        Test portfolio weight calculation.
        """
        # Calculate signals and volatility
        signals = self.strategy.calculate_momentum_signals(self.data)
        volatility = self.strategy.calculate_volatility(self.data)
        
        # Calculate weights
        idx = 200  # Use a point well after warmup
        weights = self.strategy.calculate_weights(self.data, signals, volatility, None, idx)
        
        # Should have weights for each asset
        for asset in ['A', 'B', 'C']:
            self.assertIn(asset, weights)
            
        # Asset A should have positive weight (uptrend)
        self.assertGreater(weights['A'], 0)
        
        # Asset B should have negative weight (downtrend)
        self.assertLess(weights['B'], 0)
        
        # Total leverage should not exceed max_leverage
        total_leverage = sum(abs(w) for w in weights.values())
        self.assertLessEqual(total_leverage, self.strategy.max_leverage)
    
    def test_signal_generation(self):
        """
        Test signal generation.
        """
        # Initialize strategy with data
        self.strategy.calculate_momentum_signals(self.data)
        self.strategy.calculate_volatility(self.data)
        
        # Mock current positions
        current_positions = {'A': 0, 'B': 0, 'C': 0}
        
        # Generate signals
        idx = 200  # Use a point well after warmup
        equity = 1000000
        signals = self.strategy.generate_signals(self.data, current_positions, idx, equity)
        
        # Should generate signals
        self.assertTrue(len(signals) > 0)
        
        # For each signal, check structure
        for symbol, signal in signals.items():
            self.assertIn('direction', signal)
            self.assertIn('quantity', signal)
            self.assertIn('type', signal)
            
            # Direction should be consistent with the trend
            if symbol == 'A':
                self.assertEqual(signal['direction'], 1)  # Long for uptrend
            elif symbol == 'B':
                self.assertEqual(signal['direction'], -1)  # Short for downtrend


class TestCrossSectionalMomentum(unittest.TestCase):
    """
    Test cases for the Cross-Sectional Momentum strategy.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create sample price data for multiple assets
        dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
        
        # Create assets with different performance
        prices = {}
        for i in range(10):
            # Linear trend + random noise
            trend = np.linspace(0, (i - 5) * 50, 300)  # Different trends
            noise = np.random.normal(0, 5, 300)  # Random noise
            prices[f"ASSET{i}_Close"] = 100 + trend + noise
            
        self.data = pd.DataFrame(prices, index=dates)
        
        # Initialize the strategy
        self.strategy = CrossSectionalMomentum(
            lookback_period=60,
            long_threshold=0.8,
            short_threshold=0.2,
            volatility_lookback=63,
            target_volatility=0.10,
            rebalance_frequency=20
        )
    
    def test_initialization(self):
        """
        Test strategy initialization.
        """
        self.assertEqual(self.strategy.lookback_period, 60)
        self.assertEqual(self.strategy.long_threshold, 0.8)
        self.assertEqual(self.strategy.short_threshold, 0.2)
        self.assertEqual(self.strategy.volatility_lookback, 63)
        self.assertEqual(self.strategy.target_volatility, 0.10)
        self.assertEqual(self.strategy.rebalance_frequency, 20)
    
    def test_asset_name_extraction(self):
        """
        Test asset name extraction from DataFrame.
        """
        asset_names = self.strategy._get_asset_names(self.data)
        
        # Should extract all assets
        for i in range(10):
            self.assertIn(f"ASSET{i}", asset_names)
        self.assertEqual(len(asset_names), 10)
    
    def test_momentum_rank_calculation(self):
        """
        Test momentum rank calculation.
        """
        # Calculate momentum ranks
        ranks = self.strategy.calculate_momentum_ranks(self.data)
        
        # Should have a ranks column
        self.assertIn('ranks', ranks.columns)
        
        # After warmup, ranks should have values for all assets
        idx = 100  # Well after warmup
        rank_row = ranks.iloc[idx]['ranks']
        
        for i in range(10):
            self.assertIn(f"ASSET{i}", rank_row.index)
            
        # Ranks should be between 0 and 1
        self.assertTrue((rank_row >= 0).all() and (rank_row <= 1).all())
        
        # Best performing assets should have highest ranks
        # Asset9 should be near the top, Asset0 near the bottom
        self.assertGreater(rank_row['ASSET9'], rank_row['ASSET0'])
    
    def test_weight_calculation(self):
        """
        Test portfolio weight calculation.
        """
        # Calculate ranks and volatility
        ranks = self.strategy.calculate_momentum_ranks(self.data)
        volatility = self.strategy.calculate_volatility(self.data)
        
        # Calculate weights
        idx = 100  # Well after warmup
        weights = self.strategy.calculate_weights(self.data, ranks, volatility, idx)
        
        # Should have weights for all assets
        for i in range(10):
            self.assertIn(f"ASSET{i}", weights)
            
        # Best performers (ASSET8, ASSET9) should have positive weights
        self.assertGreaterEqual(weights['ASSET9'], 0)
        
        # Worst performers (ASSET0, ASSET1) should have negative weights
        self.assertLessEqual(weights['ASSET0'], 0)
        
        # Total leverage should not exceed max_leverage (default: 1.0)
        total_leverage = sum(abs(w) for w in weights.values())
        self.assertLessEqual(total_leverage, 1.0)


class TestHybridMomentumStrategy(unittest.TestCase):
    """
    Test cases for the Hybrid Momentum Strategy.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create sample price data
        dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
        
        # Create assets with different performance
        prices = {}
        for i in range(5):
            # Linear trend + random noise
            trend = np.linspace(0, (i - 2) * 50, 300)  # Different trends
            noise = np.random.normal(0, 5, 300)  # Random noise
            prices[f"ASSET{i}_Close"] = 100 + trend + noise
            
        self.data = pd.DataFrame(prices, index=dates)
        
        # Initialize the strategy
        self.strategy = HybridMomentumStrategy(
            ts_weight=0.5,
            cs_weight=0.5,
            dynamic_allocation=True
        )
    
    def test_initialization(self):
        """
        Test strategy initialization.
        """
        self.assertEqual(self.strategy.ts_weight, 0.5)
        self.assertEqual(self.strategy.cs_weight, 0.5)
        self.assertTrue(self.strategy.dynamic_allocation)
        self.assertIsInstance(self.strategy.ts_strategy, TimeSeriesMomentum)
        self.assertIsInstance(self.strategy.cs_strategy, CrossSectionalMomentum)
    
    def test_signal_generation(self):
        """
        Test signal generation from both components.
        """
        # Initialize time-series and cross-sectional strategies
        self.strategy.ts_strategy.calculate_momentum_signals(self.data)
        self.strategy.ts_strategy.calculate_volatility(self.data)
        self.strategy.cs_strategy.calculate_momentum_ranks(self.data)
        self.strategy.cs_strategy.calculate_volatility(self.data)
        
        # Mock current positions
        current_positions = {f"ASSET{i}": 0 for i in range(5)}
        
        # Generate signals
        idx = 200  # Well after warmup
        equity = 1000000
        signals = self.strategy.generate_signals(self.data, current_positions, idx, equity)
        
        # Should generate signals
        self.assertTrue(len(signals) > 0)
        
        # For each signal, check structure
        for symbol, signal in signals.items():
            self.assertIn('direction', signal)
            self.assertIn('quantity', signal)
            self.assertIn('type', signal)
            
            # Should have component information
            if signal['type'].startswith('hybrid'):
                self.assertIn('ts_component', signal)
                self.assertIn('cs_component', signal)
    
    def test_performance_tracking_update(self):
        """
        Test performance tracking.
        """
        # Update performance metrics
        self.strategy.update_performance_tracking(0.05, 0.03)
        
        # Should record returns for both components
        self.assertEqual(self.strategy.component_performance['ts'][-1], 0.05)
        self.assertEqual(self.strategy.component_performance['cs'][-1], 0.03)
        
        # Add more data
        for _ in range(10):
            self.strategy.update_performance_tracking(0.02, 0.04)
            
        # Should have the right number of observations
        self.assertEqual(len(self.strategy.component_performance['ts']), 11)
        self.assertEqual(len(self.strategy.component_performance['cs']), 11)


if __name__ == '__main__':
    unittest.main()