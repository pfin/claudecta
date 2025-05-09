"""
Modern Momentum Strategies

This module implements modern momentum trading strategies including 
time-series, cross-sectional, and hybrid approaches with volatility
scaling and sophisticated risk management.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union


class TimeSeriesMomentum:
    """
    Time-Series Momentum Strategy.
    
    Implements a modern time-series momentum approach that generates
    signals based on absolute returns of individual assets over various
    lookback periods, with volatility scaling and risk management.
    """
    
    def __init__(
        self,
        lookback_periods: Union[int, List[int]] = [60, 120, 252],
        volatility_lookback: int = 63,
        target_volatility: float = 0.10,
        risk_scaling: bool = True,
        rebalance_frequency: int = 5,
        max_leverage: float = 2.0,
        min_momentum: float = 0.0,
        max_correlation: float = 0.7,
        use_trend_filter: bool = False,
        trend_filter_threshold: float = 0.05
    ):
        """
        Initialize the Time-Series Momentum strategy.
        
        Args:
            lookback_periods: Period(s) for momentum calculation (default: [60, 120, 252])
            volatility_lookback: Lookback period for volatility calculation (default: 63)
            target_volatility: Annualized volatility target (default: 10%)
            risk_scaling: Whether to scale positions by risk (default: True)
            rebalance_frequency: How often to rebalance in days (default: 5)
            max_leverage: Maximum leverage allowed (default: 2.0)
            min_momentum: Minimum momentum signal required (default: 0.0)
            max_correlation: Maximum allowed correlation between assets (default: 0.7)
            use_trend_filter: Whether to use trend filtering (default: False)
            trend_filter_threshold: Threshold for trend filter (default: 0.05)
        """
        self.lookback_periods = [lookback_periods] if isinstance(lookback_periods, int) else lookback_periods
        self.volatility_lookback = volatility_lookback
        self.target_volatility = target_volatility
        self.risk_scaling = risk_scaling
        self.rebalance_frequency = rebalance_frequency
        self.max_leverage = max_leverage
        self.min_momentum = min_momentum
        self.max_correlation = max_correlation
        self.use_trend_filter = use_trend_filter
        self.trend_filter_threshold = trend_filter_threshold
        
        # Internal tracking
        self.last_rebalance_idx = 0
        self.position_sizes = {}
        self.asset_weights = {}
    
    def calculate_momentum_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum signals for all assets.
        
        Args:
            prices: DataFrame with price data for each asset
            
        Returns:
            DataFrame with momentum signals for each asset and lookback period
        """
        signals = pd.DataFrame(index=prices.index)
        
        # Extract asset names
        assets = self._get_asset_names(prices)
        
        for asset in assets:
            asset_prices = self._get_asset_prices(prices, asset)
            
            for period in self.lookback_periods:
                # Calculate momentum as return over lookback period
                signals[f"{asset}_momentum_{period}"] = asset_prices.pct_change(periods=period)
        
        return signals
    
    def calculate_volatility(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility for all assets.
        
        Args:
            prices: DataFrame with price data for each asset
            
        Returns:
            DataFrame with volatility for each asset
        """
        volatility = pd.DataFrame(index=prices.index)
        
        # Extract asset names
        assets = self._get_asset_names(prices)
        
        for asset in assets:
            asset_prices = self._get_asset_prices(prices, asset)
            asset_returns = asset_prices.pct_change().dropna()
            
            # Calculate rolling volatility (annualized)
            vol = asset_returns.rolling(window=self.volatility_lookback).std() * np.sqrt(252)
            volatility[f"{asset}_volatility"] = vol
        
        return volatility
    
    def calculate_weights(
        self, 
        prices: pd.DataFrame, 
        signals: pd.DataFrame,
        volatility: pd.DataFrame,
        correlation_matrix: Optional[pd.DataFrame] = None,
        idx: int = -1
    ) -> Dict[str, float]:
        """
        Calculate portfolio weights based on momentum signals and volatility.
        
        Args:
            prices: DataFrame with price data for each asset
            signals: DataFrame with momentum signals
            volatility: DataFrame with volatility for each asset
            correlation_matrix: Correlation matrix between assets (optional)
            idx: Index to use for calculation (default: -1, latest)
            
        Returns:
            Dictionary of asset weights
        """
        weights = {}
        
        # Extract asset names
        assets = self._get_asset_names(prices)
        
        for asset in assets:
            # Combine momentum signals across lookback periods
            asset_signals = [signals[f"{asset}_momentum_{period}"].iloc[idx] 
                            for period in self.lookback_periods
                            if f"{asset}_momentum_{period}" in signals.columns]
            
            if not asset_signals or all(pd.isna(signal) for signal in asset_signals):
                weights[asset] = 0
                continue
                
            # Calculate average momentum signal
            avg_signal = np.nanmean(asset_signals)
            
            # Apply trend filter if enabled
            if self.use_trend_filter and abs(avg_signal) < self.trend_filter_threshold:
                weights[asset] = 0
                continue
                
            # Apply minimum momentum filter
            if abs(avg_signal) < self.min_momentum:
                weights[asset] = 0
                continue
            
            # Get asset volatility
            asset_vol = volatility[f"{asset}_volatility"].iloc[idx]
            
            if pd.isna(asset_vol) or asset_vol == 0:
                weights[asset] = 0
                continue
            
            # Calculate weight based on signal strength and volatility
            if self.risk_scaling:
                # Volatility-scaled weight
                weight = avg_signal * (self.target_volatility / asset_vol)
            else:
                # Simple directional weight based on signal
                weight = np.sign(avg_signal) * min(abs(avg_signal), 1.0)
                
            weights[asset] = weight
        
        # Apply correlation-based adjustments
        if correlation_matrix is not None:
            weights = self._adjust_for_correlation(weights, correlation_matrix)
            
        # Ensure total leverage doesn't exceed maximum
        weights = self._apply_leverage_constraint(weights)
            
        return weights
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        current_positions: Dict[str, float],
        index: int,
        equity: float
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals for the current timestamp.
        
        Args:
            prices: DataFrame with price data for each asset
            current_positions: Current positions for each asset
            index: Current index in the prices DataFrame
            equity: Current portfolio equity
            
        Returns:
            Dictionary of trading signals by asset
        """
        signals = {}
        
        # Check if it's time to rebalance
        if index - self.last_rebalance_idx >= self.rebalance_frequency:
            # Calculate momentum signals
            momentum_signals = self.calculate_momentum_signals(prices)
            
            # Calculate volatility
            volatility = self.calculate_volatility(prices)
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(prices, index)
            
            # Calculate target weights
            target_weights = self.calculate_weights(
                prices, momentum_signals, volatility, correlation_matrix, index
            )
            
            # Store weights
            self.asset_weights = target_weights
            self.last_rebalance_idx = index
            
            # Calculate position sizes based on weights
            self.position_sizes = self._calculate_position_sizes(
                target_weights, prices, index, equity
            )
        
        # Generate signals based on position differences
        assets = self._get_asset_names(prices)
        
        for asset in assets:
            # Skip if asset not in target positions
            if asset not in self.position_sizes:
                continue
                
            # Calculate target position
            target_position = self.position_sizes.get(asset, 0)
            
            # Current position
            current_position = current_positions.get(asset, 0)
            
            # Skip if no position change needed
            if abs(target_position - current_position) < 1:
                continue
                
            # Generate signal
            position_diff = int(target_position - current_position)
            
            if position_diff > 0:
                signals[asset] = {
                    'direction': 1,
                    'quantity': position_diff,
                    'type': 'momentum_long',
                    'weight': self.asset_weights.get(asset, 0)
                }
            elif position_diff < 0:
                signals[asset] = {
                    'direction': -1,
                    'quantity': abs(position_diff),
                    'type': 'momentum_short',
                    'weight': self.asset_weights.get(asset, 0)
                }
                
        return signals
    
    def _get_asset_names(self, prices: pd.DataFrame) -> List[str]:
        """
        Extract asset names from the prices DataFrame.
        
        Args:
            prices: DataFrame with price data
            
        Returns:
            List of asset names
        """
        # Check column structure
        if isinstance(prices.columns, pd.MultiIndex):
            # Multi-level columns: (asset, field)
            return list(prices.columns.levels[0])
        else:
            # Single-level columns: asset_field
            unique_prefixes = set()
            for col in prices.columns:
                if '_' in col:
                    prefix = col.split('_')[0]
                    unique_prefixes.add(prefix)
            return list(unique_prefixes)
    
    def _get_asset_prices(self, prices: pd.DataFrame, asset: str) -> pd.Series:
        """
        Extract price series for a specific asset.
        
        Args:
            prices: DataFrame with price data
            asset: Asset name
            
        Returns:
            Series of prices for the asset
        """
        # Check column structure
        if isinstance(prices.columns, pd.MultiIndex):
            # Multi-level columns: (asset, field)
            if (asset, 'Close') in prices.columns:
                return prices[(asset, 'Close')]
            else:
                return prices[(asset, 'Price')] if (asset, 'Price') in prices.columns else None
        else:
            # Single-level columns: asset_field
            if f"{asset}_Close" in prices.columns:
                return prices[f"{asset}_Close"]
            else:
                return prices[f"{asset}_Price"] if f"{asset}_Price" in prices.columns else None
    
    def _calculate_correlation_matrix(
        self, 
        prices: pd.DataFrame, 
        index: int,
        lookback: int = 252
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between assets.
        
        Args:
            prices: DataFrame with price data
            index: Current index
            lookback: Lookback period for correlation
            
        Returns:
            Correlation matrix
        """
        # Extract asset names
        assets = self._get_asset_names(prices)
        
        # Prepare returns data
        returns_data = {}
        for asset in assets:
            asset_prices = self._get_asset_prices(prices, asset)
            if asset_prices is not None:
                asset_returns = asset_prices.pct_change().dropna()
                returns_data[asset] = asset_returns
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        start_idx = max(0, index - lookback)
        if index > start_idx:
            return returns_df.iloc[start_idx:index].corr()
        else:
            # Not enough data, return identity matrix
            return pd.DataFrame(np.eye(len(assets)), index=assets, columns=assets)
    
    def _adjust_for_correlation(
        self, 
        weights: Dict[str, float], 
        correlation_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Adjust weights based on correlation structure.
        
        Args:
            weights: Dictionary of asset weights
            correlation_matrix: Correlation matrix between assets
            
        Returns:
            Adjusted weights dictionary
        """
        adjusted_weights = weights.copy()
        
        # Identify highly correlated pairs
        for asset1 in weights:
            for asset2 in weights:
                if asset1 != asset2 and asset1 in correlation_matrix.index and asset2 in correlation_matrix.columns:
                    correlation = correlation_matrix.loc[asset1, asset2]
                    
                    # If correlation exceeds threshold and both have same sign
                    if (abs(correlation) > self.max_correlation and 
                        np.sign(weights[asset1]) == np.sign(weights[asset2]) and
                        abs(weights[asset1]) > 0 and abs(weights[asset2]) > 0):
                        
                        # Reduce weight of the smaller position
                        if abs(weights[asset1]) < abs(weights[asset2]):
                            adjustment = (abs(correlation) - self.max_correlation) / (1 - self.max_correlation)
                            adjusted_weights[asset1] *= (1 - adjustment)
                        else:
                            adjustment = (abs(correlation) - self.max_correlation) / (1 - self.max_correlation)
                            adjusted_weights[asset2] *= (1 - adjustment)
                            
        return adjusted_weights
    
    def _apply_leverage_constraint(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply leverage constraints to weights.
        
        Args:
            weights: Dictionary of asset weights
            
        Returns:
            Adjusted weights dictionary
        """
        # Calculate total leverage
        total_leverage = sum(abs(weight) for weight in weights.values())
        
        # If leverage exceeds maximum, scale down all positions
        if total_leverage > self.max_leverage:
            scaling_factor = self.max_leverage / total_leverage
            return {asset: weight * scaling_factor for asset, weight in weights.items()}
            
        return weights
    
    def _calculate_position_sizes(
        self, 
        weights: Dict[str, float], 
        prices: pd.DataFrame, 
        index: int, 
        equity: float
    ) -> Dict[str, int]:
        """
        Convert weights to position sizes.
        
        Args:
            weights: Dictionary of asset weights
            prices: DataFrame with price data
            index: Current index
            equity: Current portfolio equity
            
        Returns:
            Dictionary of position sizes
        """
        position_sizes = {}
        
        for asset, weight in weights.items():
            if weight == 0:
                position_sizes[asset] = 0
                continue
                
            # Get current price
            asset_prices = self._get_asset_prices(prices, asset)
            if asset_prices is None or index >= len(asset_prices):
                position_sizes[asset] = 0
                continue
                
            current_price = asset_prices.iloc[index]
            
            if pd.isna(current_price) or current_price == 0:
                position_sizes[asset] = 0
                continue
                
            # Calculate position size
            position_value = weight * equity
            units = int(position_value / current_price)
            
            position_sizes[asset] = units
            
        return position_sizes


class CrossSectionalMomentum:
    """
    Cross-Sectional Momentum Strategy.
    
    Implements a modern cross-sectional momentum approach that ranks
    assets based on relative performance and generates signals for
    the top and bottom performers.
    """
    
    def __init__(
        self,
        lookback_period: int = 252,
        long_threshold: float = 0.8,
        short_threshold: float = 0.2,
        volatility_lookback: int = 63,
        target_volatility: float = 0.10,
        rebalance_frequency: int = 20,
        max_leverage: float = 1.0,
        use_vol_scaling: bool = True,
        sector_neutral: bool = False,
        sector_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the Cross-Sectional Momentum strategy.
        
        Args:
            lookback_period: Period for momentum calculation (default: 252)
            long_threshold: Percentile threshold for long positions (default: 0.8)
            short_threshold: Percentile threshold for short positions (default: 0.2)
            volatility_lookback: Lookback period for volatility calculation (default: 63)
            target_volatility: Annualized volatility target (default: 10%)
            rebalance_frequency: How often to rebalance in days (default: 20)
            max_leverage: Maximum leverage allowed (default: 1.0)
            use_vol_scaling: Whether to use volatility scaling (default: True)
            sector_neutral: Whether to use sector-neutral implementation (default: False)
            sector_mapping: Mapping of assets to sectors for sector-neutral approach
        """
        self.lookback_period = lookback_period
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.volatility_lookback = volatility_lookback
        self.target_volatility = target_volatility
        self.rebalance_frequency = rebalance_frequency
        self.max_leverage = max_leverage
        self.use_vol_scaling = use_vol_scaling
        self.sector_neutral = sector_neutral
        self.sector_mapping = sector_mapping or {}
        
        # Internal tracking
        self.last_rebalance_idx = 0
        self.position_sizes = {}
        self.asset_weights = {}
    
    def calculate_momentum_ranks(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum ranks for all assets.
        
        Args:
            prices: DataFrame with price data for each asset
            
        Returns:
            DataFrame with momentum ranks for each asset
        """
        ranks = pd.DataFrame(index=prices.index)
        
        # Extract asset names
        assets = self._get_asset_names(prices)
        
        # Calculate returns over lookback period
        returns = {}
        for asset in assets:
            asset_prices = self._get_asset_prices(prices, asset)
            if asset_prices is not None:
                returns[asset] = asset_prices.pct_change(periods=self.lookback_period)
        
        # Convert to DataFrame
        returns_df = pd.DataFrame(returns)
        
        # Calculate cross-sectional ranks (percentile)
        for idx in returns_df.index:
            row_data = returns_df.loc[idx]
            if not row_data.isna().all():
                ranks.loc[idx, 'ranks'] = row_data.rank(pct=True)
        
        return ranks
    
    def calculate_volatility(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility for all assets.
        
        Args:
            prices: DataFrame with price data for each asset
            
        Returns:
            DataFrame with volatility for each asset
        """
        volatility = pd.DataFrame(index=prices.index)
        
        # Extract asset names
        assets = self._get_asset_names(prices)
        
        for asset in assets:
            asset_prices = self._get_asset_prices(prices, asset)
            if asset_prices is not None:
                asset_returns = asset_prices.pct_change().dropna()
                
                # Calculate rolling volatility (annualized)
                vol = asset_returns.rolling(window=self.volatility_lookback).std() * np.sqrt(252)
                volatility[f"{asset}_volatility"] = vol
        
        return volatility
    
    def calculate_weights(
        self, 
        prices: pd.DataFrame, 
        ranks: pd.DataFrame,
        volatility: pd.DataFrame,
        idx: int = -1
    ) -> Dict[str, float]:
        """
        Calculate portfolio weights based on momentum ranks.
        
        Args:
            prices: DataFrame with price data for each asset
            ranks: DataFrame with momentum ranks
            volatility: DataFrame with volatility for each asset
            idx: Index to use for calculation (default: -1, latest)
            
        Returns:
            Dictionary of asset weights
        """
        weights = {}
        
        # Extract asset names
        assets = self._get_asset_names(prices)
        
        if self.sector_neutral and self.sector_mapping:
            # Sector-neutral implementation
            weights = self._calculate_sector_neutral_weights(
                assets, ranks, volatility, idx
            )
        else:
            # Standard implementation
            weights = self._calculate_standard_weights(
                assets, ranks, volatility, idx
            )
            
        # Apply leverage constraint
        weights = self._apply_leverage_constraint(weights)
            
        return weights
    
    def _calculate_standard_weights(
        self, 
        assets: List[str], 
        ranks: pd.DataFrame,
        volatility: pd.DataFrame,
        idx: int
    ) -> Dict[str, float]:
        """
        Calculate weights using standard (non-sector-neutral) approach.
        
        Args:
            assets: List of asset names
            ranks: DataFrame with momentum ranks
            volatility: DataFrame with volatility for each asset
            idx: Index to use for calculation
            
        Returns:
            Dictionary of asset weights
        """
        weights = {}
        
        if 'ranks' not in ranks.columns or idx >= len(ranks):
            return {asset: 0 for asset in assets}
            
        # Get current ranks
        current_ranks = ranks.loc[ranks.index[idx], 'ranks']
        
        # Identify long and short positions
        long_assets = []
        short_assets = []
        
        for asset in assets:
            if asset in current_ranks and not pd.isna(current_ranks[asset]):
                if current_ranks[asset] >= self.long_threshold:
                    long_assets.append(asset)
                elif current_ranks[asset] <= self.short_threshold:
                    short_assets.append(asset)
        
        # Initial equal weights
        total_assets = len(long_assets) + len(short_assets)
        if total_assets == 0:
            return {asset: 0 for asset in assets}
            
        base_weight = 1.0 / total_assets if total_assets > 0 else 0
        
        # Assign initial weights
        for asset in assets:
            if asset in long_assets:
                weights[asset] = base_weight
            elif asset in short_assets:
                weights[asset] = -base_weight
            else:
                weights[asset] = 0
                
        # Apply volatility scaling if enabled
        if self.use_vol_scaling:
            weights = self._apply_volatility_scaling(weights, volatility, idx)
            
        return weights
    
    def _calculate_sector_neutral_weights(
        self, 
        assets: List[str], 
        ranks: pd.DataFrame,
        volatility: pd.DataFrame,
        idx: int
    ) -> Dict[str, float]:
        """
        Calculate weights using sector-neutral approach.
        
        Args:
            assets: List of asset names
            ranks: DataFrame with momentum ranks
            volatility: DataFrame with volatility for each asset
            idx: Index to use for calculation
            
        Returns:
            Dictionary of asset weights
        """
        weights = {asset: 0 for asset in assets}
        
        if 'ranks' not in ranks.columns or idx >= len(ranks):
            return weights
            
        # Get current ranks
        current_ranks = ranks.loc[ranks.index[idx], 'ranks']
        
        # Group assets by sector
        sectors = {}
        for asset in assets:
            if asset in current_ranks and not pd.isna(current_ranks[asset]):
                sector = self.sector_mapping.get(asset, 'Unknown')
                if sector not in sectors:
                    sectors[sector] = []
                sectors[sector].append((asset, current_ranks[asset]))
        
        # Process each sector
        for sector, asset_ranks in sectors.items():
            # Sort assets by rank
            asset_ranks.sort(key=lambda x: x[1])
            
            # Determine long and short assets
            n_assets = len(asset_ranks)
            n_long = max(1, int(n_assets * (1 - self.long_threshold)))
            n_short = max(1, int(n_assets * self.short_threshold))
            
            # Assign equal weights within each sector
            sector_weight = 1.0 / len(sectors) if len(sectors) > 0 else 0
            asset_weight = sector_weight / max(n_long, n_short)
            
            # Long positions (top ranked)
            for i in range(n_assets - n_long, n_assets):
                if i >= 0 and i < len(asset_ranks):
                    asset = asset_ranks[i][0]
                    weights[asset] = asset_weight
            
            # Short positions (bottom ranked)
            for i in range(n_short):
                if i < len(asset_ranks):
                    asset = asset_ranks[i][0]
                    weights[asset] = -asset_weight
                    
        # Apply volatility scaling if enabled
        if self.use_vol_scaling:
            weights = self._apply_volatility_scaling(weights, volatility, idx)
            
        return weights
    
    def _apply_volatility_scaling(
        self, 
        weights: Dict[str, float],
        volatility: pd.DataFrame,
        idx: int
    ) -> Dict[str, float]:
        """
        Apply volatility scaling to weights.
        
        Args:
            weights: Dictionary of asset weights
            volatility: DataFrame with volatility for each asset
            idx: Index to use for calculation
            
        Returns:
            Volatility-scaled weights
        """
        scaled_weights = {}
        
        # Calculate inverse volatility weights
        for asset, weight in weights.items():
            if weight == 0:
                scaled_weights[asset] = 0
                continue
                
            vol_col = f"{asset}_volatility"
            if vol_col in volatility.columns and idx < len(volatility):
                vol = volatility[vol_col].iloc[idx]
                
                if not pd.isna(vol) and vol > 0:
                    # Scale weight by target vol / asset vol
                    # Preserve sign of original weight
                    scaled_weights[asset] = (self.target_volatility / vol) * np.sign(weight)
                else:
                    scaled_weights[asset] = 0
            else:
                scaled_weights[asset] = 0
                
        # Rescale to maintain original leverage
        orig_leverage = sum(abs(w) for w in weights.values())
        scaled_leverage = sum(abs(w) for w in scaled_weights.values())
        
        if scaled_leverage > 0:
            scaling_factor = orig_leverage / scaled_leverage
            return {asset: weight * scaling_factor for asset, weight in scaled_weights.items()}
        
        return scaled_weights
    
    def _apply_leverage_constraint(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply leverage constraints to weights.
        
        Args:
            weights: Dictionary of asset weights
            
        Returns:
            Adjusted weights dictionary
        """
        # Calculate total leverage
        total_leverage = sum(abs(weight) for weight in weights.values())
        
        # If leverage exceeds maximum, scale down all positions
        if total_leverage > self.max_leverage:
            scaling_factor = self.max_leverage / total_leverage
            return {asset: weight * scaling_factor for asset, weight in weights.items()}
            
        return weights
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        current_positions: Dict[str, float],
        index: int,
        equity: float
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals for the current timestamp.
        
        Args:
            prices: DataFrame with price data for each asset
            current_positions: Current positions for each asset
            index: Current index in the prices DataFrame
            equity: Current portfolio equity
            
        Returns:
            Dictionary of trading signals by asset
        """
        signals = {}
        
        # Check if it's time to rebalance
        if index - self.last_rebalance_idx >= self.rebalance_frequency:
            # Calculate momentum ranks
            ranks = self.calculate_momentum_ranks(prices)
            
            # Calculate volatility
            volatility = self.calculate_volatility(prices)
            
            # Calculate target weights
            target_weights = self.calculate_weights(prices, ranks, volatility, index)
            
            # Store weights
            self.asset_weights = target_weights
            self.last_rebalance_idx = index
            
            # Calculate position sizes based on weights
            self.position_sizes = self._calculate_position_sizes(
                target_weights, prices, index, equity
            )
        
        # Generate signals based on position differences
        assets = self._get_asset_names(prices)
        
        for asset in assets:
            # Skip if asset not in target positions
            if asset not in self.position_sizes:
                continue
                
            # Calculate target position
            target_position = self.position_sizes.get(asset, 0)
            
            # Current position
            current_position = current_positions.get(asset, 0)
            
            # Skip if no position change needed
            if abs(target_position - current_position) < 1:
                continue
                
            # Generate signal
            position_diff = int(target_position - current_position)
            
            if position_diff > 0:
                signals[asset] = {
                    'direction': 1,
                    'quantity': position_diff,
                    'type': 'cs_momentum_long',
                    'weight': self.asset_weights.get(asset, 0)
                }
            elif position_diff < 0:
                signals[asset] = {
                    'direction': -1,
                    'quantity': abs(position_diff),
                    'type': 'cs_momentum_short',
                    'weight': self.asset_weights.get(asset, 0)
                }
                
        return signals
    
    def _get_asset_names(self, prices: pd.DataFrame) -> List[str]:
        """
        Extract asset names from the prices DataFrame.
        
        Args:
            prices: DataFrame with price data
            
        Returns:
            List of asset names
        """
        # Check column structure
        if isinstance(prices.columns, pd.MultiIndex):
            # Multi-level columns: (asset, field)
            return list(prices.columns.levels[0])
        else:
            # Single-level columns: asset_field
            unique_prefixes = set()
            for col in prices.columns:
                if '_' in col:
                    prefix = col.split('_')[0]
                    unique_prefixes.add(prefix)
            return list(unique_prefixes)
    
    def _get_asset_prices(self, prices: pd.DataFrame, asset: str) -> pd.Series:
        """
        Extract price series for a specific asset.
        
        Args:
            prices: DataFrame with price data
            asset: Asset name
            
        Returns:
            Series of prices for the asset
        """
        # Check column structure
        if isinstance(prices.columns, pd.MultiIndex):
            # Multi-level columns: (asset, field)
            if (asset, 'Close') in prices.columns:
                return prices[(asset, 'Close')]
            else:
                return prices[(asset, 'Price')] if (asset, 'Price') in prices.columns else None
        else:
            # Single-level columns: asset_field
            if f"{asset}_Close" in prices.columns:
                return prices[f"{asset}_Close"]
            else:
                return prices[f"{asset}_Price"] if f"{asset}_Price" in prices.columns else None
    
    def _calculate_position_sizes(
        self, 
        weights: Dict[str, float], 
        prices: pd.DataFrame, 
        index: int, 
        equity: float
    ) -> Dict[str, int]:
        """
        Convert weights to position sizes.
        
        Args:
            weights: Dictionary of asset weights
            prices: DataFrame with price data
            index: Current index
            equity: Current portfolio equity
            
        Returns:
            Dictionary of position sizes
        """
        position_sizes = {}
        
        for asset, weight in weights.items():
            if weight == 0:
                position_sizes[asset] = 0
                continue
                
            # Get current price
            asset_prices = self._get_asset_prices(prices, asset)
            if asset_prices is None or index >= len(asset_prices):
                position_sizes[asset] = 0
                continue
                
            current_price = asset_prices.iloc[index]
            
            if pd.isna(current_price) or current_price == 0:
                position_sizes[asset] = 0
                continue
                
            # Calculate position size
            position_value = weight * equity
            units = int(position_value / current_price)
            
            position_sizes[asset] = units
            
        return position_sizes


class HybridMomentumStrategy:
    """
    Hybrid Momentum Strategy.
    
    Combines time-series and cross-sectional momentum approaches with
    dynamic allocation between the two based on market conditions.
    """
    
    def __init__(
        self,
        ts_weight: float = 0.5,
        cs_weight: float = 0.5,
        dynamic_allocation: bool = True,
        ts_strategy_params: Optional[Dict[str, Any]] = None,
        cs_strategy_params: Optional[Dict[str, Any]] = None,
        rebalance_frequency: int = 20,
        max_leverage: float = 1.5
    ):
        """
        Initialize the Hybrid Momentum strategy.
        
        Args:
            ts_weight: Initial weight for time-series component (default: 0.5)
            cs_weight: Initial weight for cross-sectional component (default: 0.5)
            dynamic_allocation: Whether to dynamically adjust weights (default: True)
            ts_strategy_params: Parameters for time-series strategy
            cs_strategy_params: Parameters for cross-sectional strategy
            rebalance_frequency: How often to rebalance in days (default: 20)
            max_leverage: Maximum leverage allowed (default: 1.5)
        """
        self.ts_weight = ts_weight
        self.cs_weight = cs_weight
        self.dynamic_allocation = dynamic_allocation
        self.rebalance_frequency = rebalance_frequency
        self.max_leverage = max_leverage
        
        # Initialize strategies
        self.ts_strategy = TimeSeriesMomentum(**(ts_strategy_params or {}))
        self.cs_strategy = CrossSectionalMomentum(**(cs_strategy_params or {}))
        
        # Internal tracking
        self.last_rebalance_idx = 0
        self.component_performance = {
            'ts': [],
            'cs': []
        }
    
    def update_allocation_weights(self, prices: pd.DataFrame, index: int):
        """
        Update allocation weights between strategy components.
        
        Args:
            prices: DataFrame with price data
            index: Current index in the data
        """
        if not self.dynamic_allocation:
            return
            
        # Check if we have enough performance history
        if len(self.component_performance['ts']) < 60 or len(self.component_performance['cs']) < 60:
            return
            
        # Calculate recent performance (last 60 days)
        ts_recent = np.mean(self.component_performance['ts'][-60:])
        cs_recent = np.mean(self.component_performance['cs'][-60:])
        
        # Calculate relative performance
        total_performance = abs(ts_recent) + abs(cs_recent)
        
        if total_performance > 0:
            # Assign weights based on relative performance
            self.ts_weight = max(0.2, min(0.8, abs(ts_recent) / total_performance))
            self.cs_weight = 1 - self.ts_weight
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        current_positions: Dict[str, float],
        index: int,
        equity: float
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals for the current timestamp.
        
        Args:
            prices: DataFrame with price data for each asset
            current_positions: Current positions for each asset
            index: Current index in the prices DataFrame
            equity: Current portfolio equity
            
        Returns:
            Dictionary of trading signals by asset
        """
        signals = {}
        
        # Check if it's time to rebalance allocation weights
        if index - self.last_rebalance_idx >= self.rebalance_frequency:
            self.update_allocation_weights(prices, index)
            self.last_rebalance_idx = index
        
        # Generate signals from each component strategy
        ts_equity = equity * self.ts_weight
        cs_equity = equity * self.cs_weight
        
        # Split current positions proportionally
        ts_positions = {asset: pos * self.ts_weight for asset, pos in current_positions.items()}
        cs_positions = {asset: pos * self.cs_weight for asset, pos in current_positions.items()}
        
        # Get signals from each strategy
        ts_signals = self.ts_strategy.generate_signals(prices, ts_positions, index, ts_equity)
        cs_signals = self.cs_strategy.generate_signals(prices, cs_positions, index, cs_equity)
        
        # Combine signals
        all_assets = set(ts_signals.keys()) | set(cs_signals.keys())
        
        for asset in all_assets:
            ts_signal = ts_signals.get(asset, {'direction': 0, 'quantity': 0, 'type': None})
            cs_signal = cs_signals.get(asset, {'direction': 0, 'quantity': 0, 'type': None})
            
            # Combine signals based on direction
            if ts_signal['direction'] == cs_signal['direction'] and ts_signal['direction'] != 0:
                # Both strategies agree on direction
                signals[asset] = {
                    'direction': ts_signal['direction'],
                    'quantity': ts_signal['quantity'] + cs_signal['quantity'],
                    'type': 'hybrid_momentum',
                    'ts_component': ts_signal['quantity'],
                    'cs_component': cs_signal['quantity']
                }
            else:
                # Strategies disagree or one is neutral
                if abs(ts_signal['quantity']) > abs(cs_signal['quantity']):
                    # Time-series signal dominates
                    if ts_signal['quantity'] != 0:
                        signals[asset] = {
                            'direction': ts_signal['direction'],
                            'quantity': ts_signal['quantity'],
                            'type': 'hybrid_ts_dominated',
                            'ts_component': ts_signal['quantity'],
                            'cs_component': 0
                        }
                elif cs_signal['quantity'] != 0:
                    # Cross-sectional signal dominates
                    signals[asset] = {
                        'direction': cs_signal['direction'],
                        'quantity': cs_signal['quantity'],
                        'type': 'hybrid_cs_dominated',
                        'ts_component': 0,
                        'cs_component': cs_signal['quantity']
                    }
        
        # Apply leverage limit
        signals = self._apply_leverage_limit(signals, equity)
            
        return signals
    
    def _apply_leverage_limit(
        self, 
        signals: Dict[str, Dict[str, Any]],
        equity: float
    ) -> Dict[str, Dict[str, Any]]:
        """
        Apply leverage limit to combined signals.
        
        Args:
            signals: Dictionary of trading signals
            equity: Current portfolio equity
            
        Returns:
            Adjusted signals dictionary
        """
        # Calculate total position value
        total_value = sum(abs(signal['quantity']) for signal in signals.values())
        
        # Check if leverage limit is exceeded
        if total_value > equity * self.max_leverage:
            # Scale down all signals proportionally
            scaling_factor = (equity * self.max_leverage) / total_value
            
            for asset in signals:
                signals[asset]['quantity'] = int(signals[asset]['quantity'] * scaling_factor)
                if 'ts_component' in signals[asset]:
                    signals[asset]['ts_component'] = int(signals[asset]['ts_component'] * scaling_factor)
                if 'cs_component' in signals[asset]:
                    signals[asset]['cs_component'] = int(signals[asset]['cs_component'] * scaling_factor)
                    
        return signals
    
    def update_performance_tracking(
        self, 
        ts_returns: float,
        cs_returns: float
    ) -> None:
        """
        Update performance tracking for strategy components.
        
        Args:
            ts_returns: Recent returns from time-series component
            cs_returns: Recent returns from cross-sectional component
        """
        self.component_performance['ts'].append(ts_returns)
        self.component_performance['cs'].append(cs_returns)
        
        # Limit history to last 252 days
        if len(self.component_performance['ts']) > 252:
            self.component_performance['ts'] = self.component_performance['ts'][-252:]
            self.component_performance['cs'] = self.component_performance['cs'][-252:]


if __name__ == "__main__":
    # Example usage
    print("Modern Momentum Strategies module")