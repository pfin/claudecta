"""
Position sizing algorithms for trading strategies.

This module implements various position sizing methods used in
trading systems, including the classic Turtle Trading ATR-based
position sizing and modern volatility-based approaches.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Union


class PositionSizer:
    """
    Base class for position sizing algorithms.
    """
    
    def __init__(self) -> None:
        """Initialize the position sizer."""
        pass
    
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        equity: float,
        **kwargs
    ) -> int:
        """
        Calculate the position size for a trade.
        
        Args:
            symbol: Trading symbol
            price: Current price
            equity: Current account equity
            **kwargs: Additional parameters
            
        Returns:
            Position size in units/shares/contracts
        """
        # Base implementation returns 0
        # Subclasses should override this method
        return 0


class FixedDollarPositionSizer(PositionSizer):
    """
    Fixed dollar amount position sizing.
    
    Allocates a fixed dollar amount to each trade.
    """
    
    def __init__(self, dollar_amount: float = 10000.0) -> None:
        """
        Initialize the fixed dollar position sizer.
        
        Args:
            dollar_amount: Dollar amount to allocate per trade
        """
        super().__init__()
        self.dollar_amount = dollar_amount
    
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        equity: float,
        **kwargs
    ) -> int:
        """
        Calculate the position size based on a fixed dollar amount.
        
        Args:
            symbol: Trading symbol
            price: Current price
            equity: Current account equity
            **kwargs: Additional parameters
            
        Returns:
            Position size in units/shares/contracts
        """
        # Ensure dollar amount doesn't exceed equity
        dollar_amount = min(self.dollar_amount, equity)
        
        # Calculate units based on price
        units = int(dollar_amount / price)
        
        return units


class FixedRiskPositionSizer(PositionSizer):
    """
    Fixed risk position sizing.
    
    Risks a fixed percentage of equity on each trade based on the distance
    to the stop loss level.
    """
    
    def __init__(self, risk_percentage: float = 0.01) -> None:
        """
        Initialize the fixed risk position sizer.
        
        Args:
            risk_percentage: Percentage of equity to risk per trade (default: 1%)
        """
        super().__init__()
        self.risk_percentage = risk_percentage
    
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        equity: float,
        stop_loss: Optional[float] = None,
        **kwargs
    ) -> int:
        """
        Calculate the position size based on fixed risk percentage.
        
        Args:
            symbol: Trading symbol
            price: Current price
            equity: Current account equity
            stop_loss: Stop loss price level
            **kwargs: Additional parameters
            
        Returns:
            Position size in units/shares/contracts
        """
        if stop_loss is None:
            # Default to 2% away from current price if no stop provided
            stop_loss = price * 0.98 if kwargs.get('direction', 1) > 0 else price * 1.02
            
        # Calculate risk per unit
        risk_per_unit = abs(price - stop_loss)
        
        if risk_per_unit == 0:
            return 0
            
        # Calculate dollar risk
        dollar_risk = equity * self.risk_percentage
        
        # Calculate units
        units = int(dollar_risk / risk_per_unit)
        
        return units


class ATRPositionSizer(PositionSizer):
    """
    ATR-based position sizing (Turtle Trading style).
    
    Uses the Average True Range to determine position size, risking
    a fixed percentage of equity per trade.
    """
    
    def __init__(
        self,
        risk_per_trade: float = 0.01,
        atr_periods: int = 20,
        atr_multiplier: float = 2.0
    ) -> None:
        """
        Initialize the ATR position sizer.
        
        Args:
            risk_per_trade: Percentage of equity to risk per trade (default: 1%)
            atr_periods: Number of periods for ATR calculation (default: 20)
            atr_multiplier: ATR multiplier for stop loss (default: 2.0)
        """
        super().__init__()
        self.risk_per_trade = risk_per_trade
        self.atr_periods = atr_periods
        self.atr_multiplier = atr_multiplier
    
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        equity: float,
        atr: Optional[float] = None,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        close: Optional[pd.Series] = None,
        index: Optional[int] = None,
        **kwargs
    ) -> int:
        """
        Calculate the position size based on ATR.
        
        Args:
            symbol: Trading symbol
            price: Current price
            equity: Current account equity
            atr: Pre-calculated ATR value (optional)
            high: Series of high prices (optional, for ATR calculation)
            low: Series of low prices (optional, for ATR calculation)
            close: Series of closing prices (optional, for ATR calculation)
            index: Current index in the data (optional, for ATR calculation)
            **kwargs: Additional parameters
            
        Returns:
            Position size in units/shares/contracts
        """
        # Calculate ATR if not provided
        if atr is None:
            if high is not None and low is not None and close is not None and index is not None:
                atr = self._calculate_atr(high, low, close, index)
            else:
                raise ValueError("Either atr or high/low/close/index must be provided")
            
        if atr == 0:
            return 0
            
        # Calculate dollar volatility (N)
        dollar_volatility = atr * self._get_point_value(symbol)
        
        # Calculate position size
        dollar_risk = equity * self.risk_per_trade
        units = int(dollar_risk / (self.atr_multiplier * dollar_volatility))
        
        return max(1, units)  # Ensure at least 1 unit
    
    def _calculate_atr(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        index: int
    ) -> float:
        """
        Calculate the Average True Range.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            index: Current index in the data
            
        Returns:
            ATR value
        """
        # Calculate True Range
        tr1 = high.iloc[index-self.atr_periods:index+1] - low.iloc[index-self.atr_periods:index+1]
        tr2 = abs(high.iloc[index-self.atr_periods:index+1] - close.iloc[index-self.atr_periods-1:index].values)
        tr3 = abs(low.iloc[index-self.atr_periods:index+1] - close.iloc[index-self.atr_periods-1:index].values)
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # Calculate ATR
        atr = tr.mean()
        
        return atr
    
    def _get_point_value(self, symbol: str) -> float:
        """
        Get the dollar value per point for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dollar value per point
        """
        # Simplified implementation
        # In a real system, this would be based on contract specifications
        return 1.0


class VolatilityScaledPositionSizer(PositionSizer):
    """
    Volatility-scaled position sizing.
    
    Modern position sizing approach that scales positions inversely
    to volatility to target a consistent level of risk.
    """
    
    def __init__(
        self,
        target_volatility: float = 0.10,
        vol_lookback: int = 63,
        max_position_size: Optional[float] = None
    ) -> None:
        """
        Initialize the volatility-scaled position sizer.
        
        Args:
            target_volatility: Annualized volatility target (default: 10%)
            vol_lookback: Lookback period for volatility calculation (default: 63 days)
            max_position_size: Maximum position size as percentage of equity (optional)
        """
        super().__init__()
        self.target_volatility = target_volatility
        self.vol_lookback = vol_lookback
        self.max_position_size = max_position_size
    
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        equity: float,
        returns: Optional[pd.Series] = None,
        volatility: Optional[float] = None,
        signal: float = 1.0,
        **kwargs
    ) -> int:
        """
        Calculate position size targeting constant volatility.
        
        Args:
            symbol: Trading symbol
            price: Current price
            equity: Current account equity
            returns: Historical returns for volatility calculation (optional)
            volatility: Pre-calculated volatility (optional)
            signal: Signal strength (-1.0 to 1.0)
            **kwargs: Additional parameters
            
        Returns:
            Position size in units/shares/contracts
        """
        # Calculate volatility if not provided
        if volatility is None:
            if returns is not None:
                volatility = self._calculate_volatility(returns)
            else:
                raise ValueError("Either volatility or returns must be provided")
        
        if volatility == 0:
            return 0
        
        # Apply minimum volatility
        volatility = max(volatility, 0.01)
        
        # Calculate position value to achieve target volatility
        # Formula: signal * (target_vol / asset_vol) * equity
        position_value = signal * (self.target_volatility / volatility) * equity
        
        # Apply maximum position size if specified
        if self.max_position_size is not None:
            max_value = equity * self.max_position_size
            position_value = min(position_value, max_value)
            
        # Convert to units
        units = int(position_value / price)
        
        return units
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """
        Calculate historical volatility for a symbol.
        
        Args:
            returns: Series of historical returns
            
        Returns:
            Annualized volatility
        """
        # Calculate standard deviation of returns
        vol = returns.iloc[-self.vol_lookback:].std()
        
        # Annualize (assuming daily data)
        vol_annualized = vol * np.sqrt(252)
        
        return vol_annualized


if __name__ == "__main__":
    # Simple example usage
    print("Position sizing module imported")