"""
Turtle Trading System Implementation

This module implements the original Turtle Trading system as developed by
Richard Dennis and William Eckhardt in the 1980s.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union

class TurtleStrategy:
    """
    Base class for Turtle Trading strategies.
    
    Implements the core functionality shared by both System 1 (short-term)
    and System 2 (long-term) versions of the Turtle Trading system.
    """
    
    def __init__(
        self,
        atr_periods: int = 20,
        risk_per_trade: float = 0.01,
        max_units: int = 4,
        pyramiding: bool = True,
        pyramiding_threshold: float = 0.5  # N/2 for additional units
    ):
        """
        Initialize the Turtle Trading strategy.
        
        Args:
            atr_periods: Number of days for ATR calculation (default: 20)
            risk_per_trade: Risk per trade as fraction of equity (default: 0.01 = 1%)
            max_units: Maximum number of units per market (default: 4)
            pyramiding: Whether to add units on favorable movement (default: True)
            pyramiding_threshold: Threshold for adding units in ATR units (default: 0.5 = N/2)
        """
        self.atr_periods = atr_periods
        self.risk_per_trade = risk_per_trade
        self.max_units = max_units
        self.pyramiding = pyramiding
        self.pyramiding_threshold = pyramiding_threshold
        
        # Internal tracking
        self.current_units = {}
        self.entry_prices = {}
        self.stops = {}
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Calculate the Average True Range (N).
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            
        Returns:
            Series of ATR values
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR (20-day simple moving average)
        atr = tr.rolling(window=self.atr_periods).mean()
        
        return atr
    
    def calculate_unit_size(
        self, 
        equity: float, 
        atr: float, 
        point_value: float = 1.0
    ) -> int:
        """
        Calculate position size for one unit based on ATR.
        
        Args:
            equity: Current account equity
            atr: Current ATR value
            point_value: Dollar value per point movement
            
        Returns:
            Unit size in contracts/shares
        """
        # Dollar volatility of one contract
        dollar_volatility = atr * point_value
        
        # Position size for 1% risk with 2N stop
        unit_size = int((equity * self.risk_per_trade) / (2 * dollar_volatility))
        
        return max(1, unit_size)  # Ensure at least 1 contract
    
    def get_breakout_levels(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        entry_lookback: int, 
        exit_lookback: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Calculate entry and exit breakout levels.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            entry_lookback: Lookback period for entry breakouts
            exit_lookback: Lookback period for exit breakouts
            
        Returns:
            Tuple of (entry_high, entry_low, exit_high, exit_low)
        """
        # Calculate entry breakout levels
        entry_high = high.rolling(window=entry_lookback).max()
        entry_low = low.rolling(window=entry_lookback).min()
        
        # Calculate exit breakout levels
        exit_high = high.rolling(window=exit_lookback).max()
        exit_low = low.rolling(window=exit_lookback).min()
        
        return entry_high, entry_low, exit_high, exit_low
    
    def generate_signals(
        self, 
        prices: pd.DataFrame, 
        position: int, 
        units: int, 
        breakout_values: Dict[str, float], 
        atr: float,
        equity: float,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Generate trading signals based on Turtle rules.
        
        Args:
            prices: DataFrame with OHLC price data
            position: Current position size
            units: Current number of units held
            breakout_values: Dictionary with breakout values
            atr: Current ATR value
            equity: Current account equity
            current_price: Current market price
            
        Returns:
            Dictionary with signal details
        """
        # Implementation to be provided by specific system subclasses
        raise NotImplementedError("Subclasses must implement generate_signals")


class TurtleSystem1(TurtleStrategy):
    """
    Turtle Trading System 1 - Short-term (20-day) breakout system.
    """
    
    def __init__(
        self,
        entry_lookback: int = 20,
        exit_lookback: int = 10,
        **kwargs
    ):
        """
        Initialize the Turtle System 1 strategy.
        
        Args:
            entry_lookback: Number of days for entry breakout (default: 20)
            exit_lookback: Number of days for exit breakout (default: 10)
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.entry_lookback = entry_lookback
        self.exit_lookback = exit_lookback
    
    def generate_signals(
        self, 
        prices: pd.DataFrame, 
        position: int, 
        units: int, 
        breakout_values: Dict[str, float], 
        atr: float,
        equity: float,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Generate trading signals based on Turtle System 1 rules.
        
        Args:
            prices: DataFrame with OHLC price data
            position: Current position size
            units: Current number of units held
            breakout_values: Dictionary with breakout values
            atr: Current ATR value
            equity: Current account equity
            current_price: Current market price
            
        Returns:
            Dictionary with signal details
        """
        signal = {
            'direction': 0,
            'quantity': 0,
            'type': None
        }
        
        # Extract breakout values
        entry_high = breakout_values['entry_high']
        entry_low = breakout_values['entry_low']
        exit_high = breakout_values['exit_high']
        exit_low = breakout_values['exit_low']
        
        # Calculate unit size
        unit_size = self.calculate_unit_size(equity, atr)
        
        # Long position logic
        if position >= 0:  # No position or long position
            # Entry signal for long
            if prices['High'].iloc[-1] > entry_high and units < self.max_units:
                signal['direction'] = 1
                signal['quantity'] = unit_size
                signal['type'] = 'entry_long'
                
                # For pyramiding - check if adding to existing position
                if position > 0 and self.pyramiding:
                    signal['type'] = 'pyramiding_long'
        
        # Short position logic
        if position <= 0:  # No position or short position
            # Entry signal for short
            if prices['Low'].iloc[-1] < entry_low and units < self.max_units:
                signal['direction'] = -1
                signal['quantity'] = unit_size
                signal['type'] = 'entry_short'
                
                # For pyramiding - check if adding to existing position
                if position < 0 and self.pyramiding:
                    signal['type'] = 'pyramiding_short'
        
        # Exit logic
        if position > 0:  # Long position
            # Exit signal for long
            if prices['Low'].iloc[-1] < exit_low:
                signal['direction'] = -1
                signal['quantity'] = abs(position)
                signal['type'] = 'exit_long'
                
            # Stop loss for long
            if 'stop' in breakout_values and prices['Low'].iloc[-1] <= breakout_values['stop']:
                signal['direction'] = -1
                signal['quantity'] = abs(position)
                signal['type'] = 'stop_loss_long'
                
        elif position < 0:  # Short position
            # Exit signal for short
            if prices['High'].iloc[-1] > exit_high:
                signal['direction'] = 1
                signal['quantity'] = abs(position)
                signal['type'] = 'exit_short'
                
            # Stop loss for short
            if 'stop' in breakout_values and prices['High'].iloc[-1] >= breakout_values['stop']:
                signal['direction'] = 1
                signal['quantity'] = abs(position)
                signal['type'] = 'stop_loss_short'
                
        return signal
    
    def calculate_stop_price(self, entry_price: float, atr: float, is_long: bool) -> float:
        """
        Calculate the stop loss price based on the 2N rule.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
            is_long: Whether the position is long
            
        Returns:
            Stop loss price
        """
        if is_long:
            return entry_price - (2 * atr)
        else:
            return entry_price + (2 * atr)


class TurtleSystem2(TurtleStrategy):
    """
    Turtle Trading System 2 - Long-term (55-day) breakout system.
    """
    
    def __init__(
        self,
        entry_lookback: int = 55,
        exit_lookback: int = 20,
        **kwargs
    ):
        """
        Initialize the Turtle System 2 strategy.
        
        Args:
            entry_lookback: Number of days for entry breakout (default: 55)
            exit_lookback: Number of days for exit breakout (default: 20)
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.entry_lookback = entry_lookback
        self.exit_lookback = exit_lookback
    
    def generate_signals(
        self, 
        prices: pd.DataFrame, 
        position: int, 
        units: int, 
        breakout_values: Dict[str, float], 
        atr: float,
        equity: float,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Generate trading signals based on Turtle System 2 rules.
        
        Args:
            prices: DataFrame with OHLC price data
            position: Current position size
            units: Current number of units held
            breakout_values: Dictionary with breakout values
            atr: Current ATR value
            equity: Current account equity
            current_price: Current market price
            
        Returns:
            Dictionary with signal details
        """
        # The implementation is similar to System 1, but with different lookback periods
        # We leverage the same logic with different parameters
        signal = {
            'direction': 0,
            'quantity': 0,
            'type': None
        }
        
        # Extract breakout values
        entry_high = breakout_values['entry_high']
        entry_low = breakout_values['entry_low']
        exit_high = breakout_values['exit_high']
        exit_low = breakout_values['exit_low']
        
        # Calculate unit size
        unit_size = self.calculate_unit_size(equity, atr)
        
        # Long position logic
        if position >= 0:  # No position or long position
            # Entry signal for long
            if prices['High'].iloc[-1] > entry_high and units < self.max_units:
                signal['direction'] = 1
                signal['quantity'] = unit_size
                signal['type'] = 'entry_long'
                
                # For pyramiding - check if adding to existing position
                if position > 0 and self.pyramiding:
                    signal['type'] = 'pyramiding_long'
        
        # Short position logic
        if position <= 0:  # No position or short position
            # Entry signal for short
            if prices['Low'].iloc[-1] < entry_low and units < self.max_units:
                signal['direction'] = -1
                signal['quantity'] = unit_size
                signal['type'] = 'entry_short'
                
                # For pyramiding - check if adding to existing position
                if position < 0 and self.pyramiding:
                    signal['type'] = 'pyramiding_short'
        
        # Exit logic
        if position > 0:  # Long position
            # Exit signal for long
            if prices['Low'].iloc[-1] < exit_low:
                signal['direction'] = -1
                signal['quantity'] = abs(position)
                signal['type'] = 'exit_long'
                
            # Stop loss for long
            if 'stop' in breakout_values and prices['Low'].iloc[-1] <= breakout_values['stop']:
                signal['direction'] = -1
                signal['quantity'] = abs(position)
                signal['type'] = 'stop_loss_long'
                
        elif position < 0:  # Short position
            # Exit signal for short
            if prices['High'].iloc[-1] > exit_high:
                signal['direction'] = 1
                signal['quantity'] = abs(position)
                signal['type'] = 'exit_short'
                
            # Stop loss for short
            if 'stop' in breakout_values and prices['High'].iloc[-1] >= breakout_values['stop']:
                signal['direction'] = 1
                signal['quantity'] = abs(position)
                signal['type'] = 'stop_loss_short'
                
        return signal
    
    def calculate_stop_price(self, entry_price: float, atr: float, is_long: bool) -> float:
        """
        Calculate the stop loss price based on the 2N rule.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
            is_long: Whether the position is long
            
        Returns:
            Stop loss price
        """
        if is_long:
            return entry_price - (2 * atr)
        else:
            return entry_price + (2 * atr)


class DualTurtleSystem:
    """
    Combined Turtle Trading System (80% System 2, 20% System 1).
    
    This follows the typical Turtle portfolio allocation approach,
    where 80% of capital is allocated to the longer-term System 2,
    and 20% to the shorter-term System 1.
    """
    
    def __init__(
        self,
        system1_allocation: float = 0.2,
        system2_allocation: float = 0.8,
        **kwargs
    ):
        """
        Initialize the Dual Turtle System strategy.
        
        Args:
            system1_allocation: Percentage of capital for System 1 (default: 0.2)
            system2_allocation: Percentage of capital for System 2 (default: 0.8)
            **kwargs: Additional arguments passed to both systems
        """
        self.system1_allocation = system1_allocation
        self.system2_allocation = system2_allocation
        
        # Initialize both systems
        self.system1 = TurtleSystem1(**kwargs)
        self.system2 = TurtleSystem2(**kwargs)
    
    def generate_signals(
        self, 
        prices: pd.DataFrame, 
        position: int, 
        units: Dict[str, int], 
        breakout_values: Dict[str, Dict[str, float]], 
        atr: float,
        equity: float,
        current_price: float
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals for both systems.
        
        Args:
            prices: DataFrame with OHLC price data
            position: Current position size
            units: Dictionary with units for each system
            breakout_values: Dictionary with breakout values for each system
            atr: Current ATR value
            equity: Current account equity
            current_price: Current market price
            
        Returns:
            Dictionary with signals for each system
        """
        # Calculate allocated equity for each system
        system1_equity = equity * self.system1_allocation
        system2_equity = equity * self.system2_allocation
        
        # Get position and units for each system
        system1_position = position * self.system1_allocation if position else 0
        system2_position = position * self.system2_allocation if position else 0
        
        system1_units = units.get('system1', 0)
        system2_units = units.get('system2', 0)
        
        # Generate signals for each system
        system1_signal = self.system1.generate_signals(
            prices, system1_position, system1_units, 
            breakout_values['system1'], atr, system1_equity, current_price
        )
        
        system2_signal = self.system2.generate_signals(
            prices, system2_position, system2_units, 
            breakout_values['system2'], atr, system2_equity, current_price
        )
        
        return {
            'system1': system1_signal,
            'system2': system2_signal
        }


if __name__ == "__main__":
    # Example usage
    print("Turtle Trading System module")