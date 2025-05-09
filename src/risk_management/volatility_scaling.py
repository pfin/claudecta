"""
Volatility scaling module for momentum strategies.

This module provides implementations of volatility scaling techniques
used in modern momentum strategies to improve risk-adjusted returns.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Union, Any


class VolatilityScaler:
    """
    Base class for volatility scaling methods.
    """
    
    def __init__(self) -> None:
        """Initialize the volatility scaler."""
        pass
    
    def scale_position(
        self,
        symbol: str,
        signal: float,
        price: float,
        volatility: float,
        equity: float,
        **kwargs
    ) -> float:
        """
        Scale a position based on volatility.
        
        Args:
            symbol: Trading symbol
            signal: Raw trading signal (-1 to 1)
            price: Current price
            volatility: Volatility metric
            equity: Current equity
            **kwargs: Additional parameters
            
        Returns:
            Scaled position size
        """
        # Base implementation returns unscaled position
        # Subclasses should override this method
        return signal * equity / price


class ConstantVolatilityTargeting(VolatilityScaler):
    """
    Constant volatility targeting for position sizing.
    
    Scales positions inversely to asset volatility to target
    a constant level of risk across all trades.
    """
    
    def __init__(
        self,
        target_volatility: float = 0.10,
        max_leverage: float = 2.0,
        min_volatility: float = 0.01,
        risk_scaling_factor: float = 1.0
    ) -> None:
        """
        Initialize the constant volatility targeting scaler.
        
        Args:
            target_volatility: Annualized volatility target (default: 10%)
            max_leverage: Maximum allowed leverage (default: 2.0)
            min_volatility: Minimum allowed volatility (default: 1%)
            risk_scaling_factor: Scaling factor for risk adjustment (default: 1.0)
        """
        super().__init__()
        self.target_volatility = target_volatility
        self.max_leverage = max_leverage
        self.min_volatility = min_volatility
        self.risk_scaling_factor = risk_scaling_factor
    
    def scale_position(
        self,
        symbol: str,
        signal: float,
        price: float,
        volatility: float,
        equity: float,
        **kwargs
    ) -> float:
        """
        Scale a position to target constant volatility.
        
        Args:
            symbol: Trading symbol
            signal: Raw trading signal (-1 to 1)
            price: Current price
            volatility: Annualized volatility (e.g., 0.20 for 20%)
            equity: Current equity
            **kwargs: Additional parameters
            
        Returns:
            Scaled position size
        """
        # Check for invalid volatility
        if pd.isna(volatility) or volatility <= 0:
            volatility = self.min_volatility
        elif volatility < self.min_volatility:
            volatility = self.min_volatility
            
        # Calculate position size to achieve target volatility
        # Formula: signal * (target_vol / asset_vol) * equity / price
        position_value = signal * (self.target_volatility / volatility) * equity
        
        # Apply risk scaling factor (useful for adjusting overall risk)
        position_value *= self.risk_scaling_factor
        
        # Apply leverage constraint
        max_position_value = equity * self.max_leverage
        position_value = np.clip(position_value, -max_position_value, max_position_value)
        
        # Convert to units
        units = position_value / price
        
        return units


class DynamicVolatilityTargeting(VolatilityScaler):
    """
    Dynamic volatility targeting for position sizing.
    
    Scales positions based on both historical volatility and
    predicted volatility, adjusting risk dynamically based on
    market conditions.
    """
    
    def __init__(
        self,
        base_target_vol: float = 0.10,
        max_leverage: float = 2.0,
        min_volatility: float = 0.01,
        vol_lookback: int = 63,
        regime_detection: str = 'simple',  # 'simple', 'adaptive', or 'garch'
        vol_regime_thresholds: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Initialize the dynamic volatility targeting scaler.
        
        Args:
            base_target_vol: Base target annualized volatility (default: 10%)
            max_leverage: Maximum allowed leverage (default: 2.0)
            min_volatility: Minimum allowed volatility (default: 1%)
            vol_lookback: Lookback period for volatility calculation (default: 63)
            regime_detection: Method for regime detection (default: 'simple')
            vol_regime_thresholds: Thresholds for volatility regimes (optional)
        """
        super().__init__()
        self.base_target_vol = base_target_vol
        self.max_leverage = max_leverage
        self.min_volatility = min_volatility
        self.vol_lookback = vol_lookback
        self.regime_detection = regime_detection
        
        # Default regime thresholds
        self.vol_regime_thresholds = vol_regime_thresholds or {
            'low': 0.10,      # Low volatility threshold
            'high': 0.20,     # High volatility threshold
            'extreme': 0.30   # Extreme volatility threshold
        }
        
        # Default regime adjustment factors
        self.regime_adjustment_factors = {
            'low': 1.2,       # Increase exposure in low vol
            'normal': 1.0,    # Normal exposure
            'high': 0.8,      # Reduce exposure in high vol
            'extreme': 0.5    # Significantly reduce in extreme vol
        }
        
        # Internal state
        self.vol_history = {}
    
    def update_vol_history(self, symbol: str, volatility: float) -> None:
        """
        Update volatility history for a symbol.
        
        Args:
            symbol: Trading symbol
            volatility: Current volatility estimate
        """
        if symbol not in self.vol_history:
            self.vol_history[symbol] = []
            
        self.vol_history[symbol].append(volatility)
        
        # Limit history length
        max_history = 252  # Keep about 1 year of data
        if len(self.vol_history[symbol]) > max_history:
            self.vol_history[symbol] = self.vol_history[symbol][-max_history:]
    
    def detect_vol_regime(self, symbol: str, current_vol: float) -> str:
        """
        Detect current volatility regime.
        
        Args:
            symbol: Trading symbol
            current_vol: Current volatility estimate
            
        Returns:
            Volatility regime ('low', 'normal', 'high', or 'extreme')
        """
        if self.regime_detection == 'adaptive' and symbol in self.vol_history and len(self.vol_history[symbol]) > 63:
            # Use historical data to determine adaptive thresholds
            vol_series = pd.Series(self.vol_history[symbol])
            thresholds = {
                'low': vol_series.quantile(0.25),
                'high': vol_series.quantile(0.75),
                'extreme': vol_series.quantile(0.9)
            }
        else:
            # Use default thresholds
            thresholds = self.vol_regime_thresholds
        
        # Determine current regime
        if current_vol <= thresholds['low']:
            return 'low'
        elif current_vol >= thresholds['extreme']:
            return 'extreme'
        elif current_vol >= thresholds['high']:
            return 'high'
        else:
            return 'normal'
    
    def scale_position(
        self,
        symbol: str,
        signal: float,
        price: float,
        volatility: float,
        equity: float,
        returns: Optional[pd.Series] = None,
        **kwargs
    ) -> float:
        """
        Scale a position using dynamic volatility targeting.
        
        Args:
            symbol: Trading symbol
            signal: Raw trading signal (-1 to 1)
            price: Current price
            volatility: Annualized volatility (e.g., 0.20 for 20%)
            equity: Current equity
            returns: Historical returns (optional, for GARCH forecasting)
            **kwargs: Additional parameters
            
        Returns:
            Scaled position size
        """
        # Check for invalid volatility
        if pd.isna(volatility) or volatility <= 0:
            volatility = self.min_volatility
        elif volatility < self.min_volatility:
            volatility = self.min_volatility
            
        # Update volatility history
        self.update_vol_history(symbol, volatility)
        
        # Detect volatility regime
        regime = self.detect_vol_regime(symbol, volatility)
        
        # Get regime adjustment factor
        adjustment_factor = self.regime_adjustment_factors.get(regime, 1.0)
        
        # Calculate adjusted target volatility
        target_vol = self.base_target_vol * adjustment_factor
        
        # Forecast volatility using GARCH if applicable
        if self.regime_detection == 'garch' and returns is not None and len(returns) > 100:
            try:
                # Optional GARCH implementation (requires arch package)
                from arch import arch_model
                model = arch_model(returns * 100, vol='GARCH', p=1, q=1)
                result = model.fit(disp='off')
                forecast = result.forecast(horizon=1)
                garch_vol = np.sqrt(forecast.variance.iloc[-1, 0]) / 100 * np.sqrt(252)
                
                # Blend historical and GARCH volatility
                volatility = 0.7 * volatility + 0.3 * garch_vol
            except (ImportError, Exception):
                # Fallback to historical volatility if GARCH fails
                pass
        
        # Calculate position size to achieve target volatility
        # Formula: signal * (target_vol / volatility) * equity / price
        position_value = signal * (target_vol / volatility) * equity
        
        # Apply leverage constraint
        max_position_value = equity * self.max_leverage
        position_value = np.clip(position_value, -max_position_value, max_position_value)
        
        # Convert to units
        units = position_value / price
        
        return units


if __name__ == "__main__":
    # Simple example usage
    print("Volatility scaling module imported")