"""
Unit tests for risk management components.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import risk management components to test
from src.risk_management.position_sizing import (
    PositionSizer, FixedDollarPositionSizer, FixedRiskPositionSizer, 
    ATRPositionSizer, VolatilityScaledPositionSizer
)
from src.risk_management.volatility_scaling import (
    VolatilityScaler, ConstantVolatilityTargeting, DynamicVolatilityTargeting
)


class TestPositionSizing(unittest.TestCase):
    """
    Test cases for position sizing algorithms.
    """
    
    def test_base_position_sizer(self):
        """
        Test the base PositionSizer class.
        """
        sizer = PositionSizer()
        
        # Base implementation should return 0
        position_size = sizer.calculate_position_size("TEST", 100.0, 10000.0)
        self.assertEqual(position_size, 0)
    
    def test_fixed_dollar_position_sizer(self):
        """
        Test the FixedDollarPositionSizer class.
        """
        # Initialize with $1000 per position
        sizer = FixedDollarPositionSizer(dollar_amount=1000.0)
        
        # Test with $100 stock price (should get 10 shares)
        position_size = sizer.calculate_position_size("TEST", 100.0, 10000.0)
        self.assertEqual(position_size, 10)
        
        # Test with high price relative to dollar amount
        position_size = sizer.calculate_position_size("TEST", 2000.0, 10000.0)
        self.assertEqual(position_size, 0)  # Can't afford even 1 share
        
        # Test with low equity (should be limited by equity)
        position_size = sizer.calculate_position_size("TEST", 100.0, 500.0)
        self.assertEqual(position_size, 5)  # Limited by equity
    
    def test_fixed_risk_position_sizer(self):
        """
        Test the FixedRiskPositionSizer class.
        """
        # Initialize with 1% risk per trade
        sizer = FixedRiskPositionSizer(risk_percentage=0.01)
        
        # Test with $100 stock and $90 stop (10% risk per share)
        position_size = sizer.calculate_position_size(
            "TEST", 100.0, 10000.0, stop_loss=90.0
        )
        self.assertEqual(position_size, 10)  # $100 risk / $10 risk per share = 10 shares
        
        # Test with smaller stop distance
        position_size = sizer.calculate_position_size(
            "TEST", 100.0, 10000.0, stop_loss=95.0
        )
        self.assertEqual(position_size, 20)  # $100 risk / $5 risk per share = 20 shares
        
        # Test with default stop calculation
        position_size = sizer.calculate_position_size(
            "TEST", 100.0, 10000.0, direction=1
        )
        self.assertTrue(position_size > 0)
    
    def test_atr_position_sizer(self):
        """
        Test the ATRPositionSizer class.
        """
        # Initialize with 1% risk per trade and 2.0 ATR multiplier
        sizer = ATRPositionSizer(risk_per_trade=0.01, atr_multiplier=2.0)
        
        # Test with pre-calculated ATR
        position_size = sizer.calculate_position_size(
            "TEST", 100.0, 10000.0, atr=5.0
        )
        
        # Expected calculation: $100 risk / (2 * $5 ATR) = 10 shares
        expected_size = int((10000.0 * 0.01) / (2.0 * 5.0))
        self.assertEqual(position_size, expected_size)
        
        # Test with zero ATR (should return 0)
        position_size = sizer.calculate_position_size(
            "TEST", 100.0, 10000.0, atr=0.0
        )
        self.assertEqual(position_size, 0)
    
    def test_volatility_scaled_position_sizer(self):
        """
        Test the VolatilityScaledPositionSizer class.
        """
        # Initialize with 10% target volatility
        sizer = VolatilityScaledPositionSizer(target_volatility=0.10)
        
        # Test with 20% asset volatility (should scale down)
        position_size = sizer.calculate_position_size(
            "TEST", 100.0, 10000.0, volatility=0.20, signal=1.0
        )
        
        # Expected: signal * (target_vol / asset_vol) * equity / price
        # = 1.0 * (0.10 / 0.20) * 10000 / 100 = 50 shares
        expected_size = int(1.0 * (0.10 / 0.20) * 10000.0 / 100.0)
        self.assertEqual(position_size, expected_size)
        
        # Test with 5% asset volatility (should scale up)
        position_size = sizer.calculate_position_size(
            "TEST", 100.0, 10000.0, volatility=0.05, signal=1.0
        )
        
        # Expected: 1.0 * (0.10 / 0.05) * 10000 / 100 = 200 shares
        expected_size = int(1.0 * (0.10 / 0.05) * 10000.0 / 100.0)
        self.assertEqual(position_size, expected_size)
        
        # Test with negative signal (short position)
        position_size = sizer.calculate_position_size(
            "TEST", 100.0, 10000.0, volatility=0.10, signal=-1.0
        )
        
        # Expected: -1.0 * (0.10 / 0.10) * 10000 / 100 = -100 shares
        expected_size = int(-1.0 * (0.10 / 0.10) * 10000.0 / 100.0)
        self.assertEqual(position_size, expected_size)


class TestVolatilityScaling(unittest.TestCase):
    """
    Test cases for volatility scaling components.
    """
    
    def test_base_volatility_scaler(self):
        """
        Test the base VolatilityScaler class.
        """
        scaler = VolatilityScaler()
        
        # Base implementation should return signal * equity / price
        scaled_size = scaler.scale_position("TEST", 1.0, 100.0, 0.10, 10000.0)
        self.assertEqual(scaled_size, 100.0)  # 1.0 * 10000 / 100
    
    def test_constant_volatility_targeting(self):
        """
        Test the ConstantVolatilityTargeting class.
        """
        # Initialize with 10% target volatility and 2.0 max leverage
        scaler = ConstantVolatilityTargeting(
            target_volatility=0.10,
            max_leverage=2.0,
            min_volatility=0.01
        )
        
        # Test with 20% asset volatility (should scale down)
        scaled_size = scaler.scale_position(
            "TEST", 1.0, 100.0, 0.20, 10000.0
        )
        
        # Expected: signal * (target_vol / asset_vol) * equity / price
        # = 1.0 * (0.10 / 0.20) * 10000 / 100 = 50.0 shares
        expected_size = 1.0 * (0.10 / 0.20) * 10000.0 / 100.0
        self.assertEqual(scaled_size, expected_size)
        
        # Test with very low volatility (should use minimum)
        scaled_size = scaler.scale_position(
            "TEST", 1.0, 100.0, 0.001, 10000.0
        )
        
        # Expected: 1.0 * (0.10 / 0.01) * 10000 / 100 = 1000.0 shares
        expected_size = 1.0 * (0.10 / 0.01) * 10000.0 / 100.0
        
        # But this would exceed max leverage, so should be capped
        expected_size = min(expected_size, 2.0 * 10000.0 / 100.0)
        self.assertEqual(scaled_size, expected_size)
        
        # Test with negative volatility (should use minimum)
        scaled_size = scaler.scale_position(
            "TEST", 1.0, 100.0, -0.10, 10000.0
        )
        expected_size = 1.0 * (0.10 / 0.01) * 10000.0 / 100.0
        expected_size = min(expected_size, 2.0 * 10000.0 / 100.0)
        self.assertEqual(scaled_size, expected_size)
    
    def test_dynamic_volatility_targeting(self):
        """
        Test the DynamicVolatilityTargeting class.
        """
        # Initialize with 10% base target volatility
        scaler = DynamicVolatilityTargeting(
            base_target_vol=0.10,
            max_leverage=2.0,
            min_volatility=0.01,
            regime_detection='simple'
        )
        
        # Test with moderate volatility (normal regime)
        scaled_size = scaler.scale_position(
            "TEST", 1.0, 100.0, 0.15, 10000.0
        )
        
        # Expected for normal regime: signal * (0.10 * 1.0) / 0.15 * 10000 / 100
        expected_size = 1.0 * (0.10 * 1.0) / 0.15 * 10000.0 / 100.0
        self.assertAlmostEqual(scaled_size, expected_size, delta=0.01)
        
        # Test with high volatility (high regime)
        scaled_size = scaler.scale_position(
            "TEST", 1.0, 100.0, 0.25, 10000.0
        )
        
        # Expected for high regime: signal * (0.10 * 0.8) / 0.25 * 10000 / 100
        # Reduced target vol by regime adjustment factor
        expected_size = 1.0 * (0.10 * 0.8) / 0.25 * 10000.0 / 100.0
        self.assertAlmostEqual(scaled_size, expected_size, delta=0.01)
        
        # Test with extreme volatility
        scaled_size = scaler.scale_position(
            "TEST", 1.0, 100.0, 0.35, 10000.0
        )
        
        # Expected for extreme regime: signal * (0.10 * 0.5) / 0.35 * 10000 / 100
        # Severely reduced target vol
        expected_size = 1.0 * (0.10 * 0.5) / 0.35 * 10000.0 / 100.0
        self.assertAlmostEqual(scaled_size, expected_size, delta=0.01)
        
        # Test with adaptive thresholds
        scaler = DynamicVolatilityTargeting(
            base_target_vol=0.10,
            max_leverage=2.0,
            min_volatility=0.01,
            regime_detection='adaptive'
        )
        
        # Update volatility history
        for vol in [0.10, 0.12, 0.15, 0.18, 0.20]:
            scaler.update_vol_history("TEST", vol)
            
        # Test with new volatility
        scaled_size = scaler.scale_position(
            "TEST", 1.0, 100.0, 0.22, 10000.0
        )
        
        # Should be a valid size
        self.assertTrue(scaled_size > 0)


if __name__ == '__main__':
    unittest.main()