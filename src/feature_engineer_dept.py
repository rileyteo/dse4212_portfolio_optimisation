from .imports import *

class FeatureEngineer:
    """
    Compute time-series and cross-sectional features from OHLC data
    """
    
    def __init__(self, 
                 open,
                 high,
                 low,
                 close,
                 volume,
                 returns,
                 risk_free_rate,
                 lookback_days: int = 60):
        """
        Args:
            ohlc_df: Long format DataFrame with columns: 
                     ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
            rf_rate: Series with date index and annualized 3-month T-bill rate (%)
            lookback_days: Minimum days of history needed for features
        """
        print("Initializing Feature Engineer...")

        self.close = close
        self.open = open
        self.high = high
        self.low = low
        self.volume = volume
        self.rf_rate = risk_free_rate 

        # Calculate returns
        self.returns = returns
        
        # Excess returns
        self.excess_returns = self.returns.sub(self.rf_rate, axis=0)
        
        self.lookback = lookback_days
        self.tickers = self.close.columns.tolist()
        self.dates = self.returns.index
        
        print(f"  Stocks: {len(self.tickers)}")
        print(f"  Date range: {self.dates[0].date()} to {self.dates[-1].date()}")
        print(f"  Lookback period: {lookback_days} days")

    def compute_all_features(self, save_path: str = None) -> pd.DataFrame:
        """
        Compute all features for all dates with sufficient history
        
        Args:
            save_path: Optional path to save features as pickle
        
        Returns:
            DataFrame with MultiIndex (date, ticker) and feature columns
        """
        print("\nComputing features...")
        
        # Only compute for dates with sufficient history
        start_idx = self.lookback
        valid_dates = self.dates[start_idx:]
        
        all_features = {}
        
        for i, date in enumerate(valid_dates):
            if i % 50 == 0 or i == len(valid_dates) - 1:
                print(f"  Progress: {i+1}/{len(valid_dates)} ({(i+1)/len(valid_dates)*100:.1f}%)")
            
            # Get data up to (but not including) this date
            date_idx = self.dates.get_loc(date)
            window_start = date_idx - self.lookback
            window_end = date_idx
            
            # Extract windows
            returns_window = self.returns.iloc[window_start:window_end]
            excess_returns_window = self.excess_returns.iloc[window_start:window_end]
            close_window = self.close.iloc[window_start:window_end]
            volume_window = self.volume.iloc[window_start:window_end]
            high_window = self.high.iloc[window_start:window_end]
            low_window = self.low.iloc[window_start:window_end]
            
            # Compute features for this date
            date_features = self._compute_features_single_date(
                returns_window, 
                excess_returns_window,
                close_window,
                volume_window,
                high_window,
                low_window
            )            

            date_features = date_features.set_index('ticker')
            all_features[date] = date_features
            if save_path:
                if not date_features.isna().values.any():
                    with open(save_path+str(date.date())+".pkl", 'wb') as f:
                        pickle.dump(date_features, f)  
                else:
                    print(date)   
            
        # Combine all dates
        features_df = pd.concat(all_features, ignore_index=True)
        
        print(f"\n✓ Feature computation complete!")
        print(f"  Shape: {features_df.shape}")
        print(f"  Features: {features_df.shape[1]}")
        print(f"  Memory: {features_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        
        return features_df
    
    def _compute_features_single_date(self,
                                     returns_window: pd.DataFrame,
                                     excess_returns_window: pd.DataFrame,
                                     close_window: pd.DataFrame,
                                     volume_window: pd.DataFrame,
                                     high_window: pd.DataFrame,
                                     low_window: pd.DataFrame,
                                     for_return: bool = True) -> pd.DataFrame:
        """
        Compute all features for one date across all stocks
        """
        n_stocks = len(self.tickers)
        features_list = []
        
        # === TIME-SERIES FEATURES (per stock) ===
        for i, ticker in enumerate(self.tickers):
            feat = {'ticker': ticker}
            
            ret = returns_window.iloc[:, i].values
            excess_ret = excess_returns_window.iloc[:, i].values
            price = close_window.iloc[:, i].values
            vol = volume_window.iloc[:, i].values
            high = high_window.iloc[:, i].values
            low = low_window.iloc[:, i].values
            
            if for_return:
                # 1. Momentum (various periods)
                feat['ret_1d'] = ret[-1] if len(ret) > 0 else 0
                feat['ret_5d'] = np.sum(ret[-5:]) if len(ret) >= 5 else 0
                feat['ret_21d'] = np.sum(ret[-21:]) if len(ret) >= 21 else 0
                feat['ret_63d'] = np.sum(ret[-63:]) if len(ret) >= 63 else 0
                
                # 2. Excess returns (risk-adjusted momentum)
                feat['excess_ret_21d'] = np.sum(excess_ret[-21:]) if len(excess_ret) >= 21 else 0
                feat['excess_ret_63d'] = np.sum(excess_ret[-63:]) if len(excess_ret) >= 63 else 0
                
                # 3. Volatility
                feat['vol_5d'] = np.std(ret[-5:]) if len(ret) >= 5 else 0
                feat['vol_21d'] = np.std(ret[-21:]) if len(ret) >= 21 else 0
                feat['vol_63d'] = np.std(ret[-63:]) if len(ret) >= 63 else 0
                
                # 4. Sharpe-like ratio (excess return / volatility)
                if len(excess_ret) >= 21 and feat['vol_21d'] > 0:
                    feat['sharpe_21d'] = feat['excess_ret_21d'] / (feat['vol_21d'] * np.sqrt(21))
                else:
                    feat['sharpe_21d'] = 0
                
                # 5. Moving average ratios
                if len(price) >= 21:
                    ma20 = np.mean(price[-21:])
                    feat['price_ma20'] = (price[-1] / ma20) - 1
                else:
                    feat['price_ma20'] = 0
                
                if len(price) >= 50:
                    ma50 = np.mean(price[-50:])
                    feat['price_ma50'] = (price[-1] / ma50) - 1
                else:
                    feat['price_ma50'] = 0
                
                # 6. Momentum acceleration (recent vs older)
                if len(ret) >= 21:
                    recent = np.sum(ret[-5:])
                    older = np.sum(ret[-21:-5]) / 16 * 5  # Normalize to same period
                    feat['momentum_accel'] = recent - older
                else:
                    feat['momentum_accel'] = 0
                
                # 7. Volume ratio
                if len(vol) >= 21:
                    avg_vol = np.mean(vol[-21:])
                    current_vol = vol[-1]
                    feat['volume_ratio'] = current_vol / avg_vol if avg_vol > 0 else 1
                else:
                    feat['volume_ratio'] = 1
                
                # 8. High-low range position
                if len(high) >= 21:
                    high_21 = np.max(high[-21:])
                    low_21 = np.min(low[-21:])
                    if high_21 != low_21:
                        feat['range_position'] = (price[-1] - low_21) / (high_21 - low_21)
                    else:
                        feat['range_position'] = 0.5
                else:
                    feat['range_position'] = 0.5
                
                # 9. Drawdown from peak
                if len(price) >= 21:
                    peak = np.max(price[-21:])
                    feat['drawdown_21d'] = (price[-1] / peak) - 1
                else:
                    feat['drawdown_21d'] = 0
                
                # 10. Skewness and kurtosis
                if len(ret) >= 21:
                    feat['skew_21d'] = stats.skew(ret[-21:])
                    feat['kurt_21d'] = stats.kurtosis(ret[-21:])
                else:
                    feat['skew_21d'] = 0
                    feat['kurt_21d'] = 0
                
                # 11. RSI 14
                if len(price) >= 15:
                    deltas = np.diff(price[-15:])
                    ups = deltas[deltas > 0].sum() / 14
                    downs = -deltas[deltas < 0].sum() / 14
                    rs = ups / downs if downs > 0 else 0
                    feat['rsi_14'] = 100 - (100 / (1 + rs))

                # 12. Lagged returns
                for i in range(1, 21):
                    feat[f'ret_lag_{i}d'] = ret[-i] if len(ret) >= i else 0
                
                # # EWMA volatility
                # if len(ret) >= 21:
                #     lambda_decay = 0.94
                #     ewma_var = 0
                #     for i in range(len(ret)-1, max(len(ret)-22, 0), -1):
                #         ewma_var = lambda_decay * ewma_var + (1 - lambda_decay) * ret[i]**2
                #     feat['vol_ewma'] = np.sqrt(ewma_var) if ewma_var > 0 else 0
                # else:
                #     feat['vol_ewma'] = 0
                
            else: # For volatility features
                # 1. Volatility
                feat['vol_5d'] = np.std(ret[-5:]) if len(ret) >= 5 else 0
                feat['vol_21d'] = np.std(ret[-21:]) if len(ret) >= 21 else 0
                feat['vol_63d'] = np.std(ret[-63:]) if len(ret) >= 63 else 0 

                # 2. Volume ratio
                if len(vol) >= 21:
                    avg_vol = np.mean(vol[-21:])
                    current_vol = vol[-1]
                    feat['volume_ratio'] = current_vol / avg_vol if avg_vol > 0 else 1
                else:
                    feat['volume_ratio'] = 1                   

                # 4. Skewness and kurtosis
                if len(ret) >= 21:
                    feat['skew_21d'] = stats.skew(ret[-21:])
                    feat['kurt_21d'] = stats.kurtosis(ret[-21:])
                else:
                    feat['skew_21d'] = 0
                    feat['kurt_21d'] = 0

                # 5 . Recent Shocks
                feat['rv_5d'] = np.mean(ret[-5:]**2) if len(ret) >= 5 else 0
                feat['rv_21d'] = np.mean(ret[-21:]**2) if len(ret) >= 21 else 0
                feat['rv_63d'] = np.mean(ret[-63:]**2) if len(ret) >= 63 else 0

                #6. Realised variance ratios
                feat['rv_ratio_5_21'] = feat['rv_5d'] / feat['rv_21d'] if feat['rv_21d'] > 0 else 0
                feat['rv_ratio_5_63'] = feat['rv_5d'] / feat['rv_63d'] if feat['rv_63d'] > 0 else 0
                feat['rv_ratio_21_63'] = feat['rv_21d'] / feat['rv_63d'] if feat['rv_63d'] > 0 else 0

                #7. Realised variance trend
                feat['rv_trend'] = (feat['rv_21d'] - feat['rv_63d']) / feat['rv_63d'] if feat['rv_63d'] > 0 else 0

                # 8. EWMA volatility
                if len(ret) >= 21:
                    lambda_decay = 0.94
                    ewma_var = 0
                    for i in range(len(ret)-1, max(len(ret)-22, 0), -1):
                        ewma_var = lambda_decay * ewma_var + (1 - lambda_decay) * ret[i]**2
                    feat['vol_ewma'] = np.sqrt(ewma_var) if ewma_var > 0 else 0
                else:
                    feat['vol_ewma'] = 0

                # 9. Volume Ratio
                if len(vol) >= 21:
                    avg_vol = np.mean(vol[-21:])
                    current_vol = vol[-1]
                    feat['volume_ratio'] = current_vol / avg_vol if avg_vol > 0 else 1
                else:
                    feat['volume_ratio'] = 1


            features_list.append(feat)
            

        
        df = pd.DataFrame(features_list)
        
        # === CROSS-SECTIONAL FEATURES (compare stocks) ===
        
        # Market (equal-weighted)
        market_ret = returns_window.mean(axis=1)
        
        if for_return:
            # 1. Percentile ranks
            for col in ['ret_21d', 'ret_63d', 'vol_21d', 'excess_ret_21d']:
                df[f'{col}_pct'] = df[col].rank(pct=True)
            
            # 2. Relative to median
            for col in ['ret_21d', 'ret_63d', 'vol_21d']:
                median = df[col].median()
                df[f'{col}_vs_med'] = df[col] - median
            
            # 3. Z-scores
            for col in ['ret_21d', 'vol_21d']:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[f'{col}_zscore'] = (df[col] - mean) / std
                else:
                    df[f'{col}_zscore'] = 0
            
            # 4. Beta to market
            betas = []
            for i in range(n_stocks):
                stock_ret = returns_window.iloc[:, i].values
                if len(stock_ret) >= 60 and len(market_ret) >= 60:
                    cov = np.cov(stock_ret[-60:], market_ret[-60:].values)[0, 1]
                    var = np.var(market_ret[-60:].values)
                    beta = cov / var if var > 0 else 1
                else:
                    beta = 1
                betas.append(beta)
            
            df['beta'] = betas
            df['beta_pct'] = pd.Series(betas).rank(pct=True).values
            
            # 5. Correlation to market
            correlations = []
            for i in range(n_stocks):
                stock_ret = returns_window.iloc[:, i].values
                if len(stock_ret) >= 60 and len(market_ret) >= 60:
                    corr = np.corrcoef(stock_ret[-60:], market_ret[-60:].values)[0, 1]
                else:
                    corr = 0
                correlations.append(corr)
            
            df['corr_market'] = correlations
            
            # 6. Risk-adjusted momentum (interaction feature)
            df['risk_adj_mom'] = df['excess_ret_21d'] / (df['vol_21d'] + 1e-8)
            df['risk_adj_mom_pct'] = df['risk_adj_mom'].rank(pct=True)
            
            # 7. Momentum-volatility interaction
            df['mom_vol_interaction'] = df['ret_21d'] * df['vol_21d']
        
            # 4. Beta to market
            betas = []
            for i in range(n_stocks):
                stock_ret = returns_window.iloc[:, i].values
                if len(stock_ret) >= 60 and len(market_ret) >= 60:
                    cov = np.cov(stock_ret[-60:], market_ret[-60:].values)[0, 1]
                    var = np.var(market_ret[-60:].values)
                    beta = cov / var if var > 0 else 1
                else:
                    beta = 1
                betas.append(beta)

            # 5. Correlation to market
            correlations = []
            for i in range(n_stocks):
                stock_ret = returns_window.iloc[:, i].values
                if len(stock_ret) >= 60 and len(market_ret) >= 60:
                    corr = np.corrcoef(stock_ret[-60:], market_ret[-60:].values)[0, 1]
                else:
                    corr = 0
                correlations.append(corr)
            
            df['corr_market'] = correlations
        
        else:
            # 1. Percentile ranks
            for col in ['vol_21d', 'vol_5d', 'vol_63d']:
                df[f'{col}_pct'] = df[col].rank(pct=True)
            
            # 2. Relative to median
            for col in ['vol_21d']:
                median = df[col].median()
                df[f'{col}_vs_med'] = df[col] - median
            
            # 3. Z-scores
            for col in ['vol_21d', 'vol_5d', 'vol_63d']:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[f'{col}_zscore'] = (df[col] - mean) / std
                else:
                    df[f'{col}_zscore'] = 0  
        return df
    
    def compute_all_features_volatility_weekly(self, save_path: str = None) -> pd.DataFrame:
        """
        Compute features for weekly volatility prediction
        
        Features computed on weekly aggregated data
        Target: Next week's realized volatility
        
        Args:
            save_path: Optional path to save features as pickle
        
        Returns:
            DataFrame with features and volatility targets
        """
        print("\nComputing weekly volatility features...")
        
        # Get weekly rebalance dates (Mondays or first trading day of week)
        weekly_dates = self._get_weekly_dates()
        
        print(f"  Weekly dates: {len(weekly_dates)}")
        print(f"  Date range: {weekly_dates[0].date()} to {weekly_dates[-1].date()}")
        
        all_features = {}
        predicted_vols = {}
        
        for i, week_date in enumerate(weekly_dates):
            if i % 10 == 0 or i == len(weekly_dates) - 1:
                print(f"  Progress: {i+1}/{len(weekly_dates)} ({(i+1)/len(weekly_dates)*100:.1f}%)")
            
            # === STEP 1: Get historical daily data (for feature calculation) ===
            date_idx = self.dates.get_loc(week_date)
            window_start_idx = max(0, date_idx - self.lookback)
            window_end_idx = date_idx
            
            # Historical daily data UP TO this week
            daily_returns_window = self.returns.iloc[window_start_idx:window_end_idx]
            daily_excess_returns_window = self.excess_returns.iloc[window_start_idx:window_end_idx]
            daily_close_window = self.close.iloc[window_start_idx:window_end_idx]
            daily_volume_window = self.volume.iloc[window_start_idx:window_end_idx]
            daily_high_window = self.high.iloc[window_start_idx:window_end_idx]
            daily_low_window = self.low.iloc[window_start_idx:window_end_idx]

            week_features = self._compute_features_single_date(
                daily_returns_window,        # Use daily for more granular vol calc
                daily_excess_returns_window,
                daily_close_window,
                daily_volume_window,
                daily_high_window,
                daily_low_window,
                for_return=False
            )
            
            # === Calculate TARGET (next week's realized volatility) ===
            # Get next week's data (5 trading days ahead)
            next_week_start_idx = date_idx + 1
            next_week_end_idx = min(date_idx + 6, len(self.returns))  # +6 to get ~5 trading days
            
            if next_week_end_idx <= next_week_start_idx:
                continue  # Skip if not enough future data
            
            # Next week's OHLC data
            next_week_returns = self.returns.iloc[next_week_start_idx:next_week_end_idx]
            
            # Calculate target volatility for each stock
            target_vol = self._calculate_target_volatility(next_week_returns)
            
            # Add target to features
            week_features = week_features.set_index('ticker')
            
            # # Also save volatility percentile (handles non-stationarity)
            # week_features['target_vol_pct'] = target_vol.rank(pct=True)
            
            all_features[week_date] = week_features
            
            # Save individual pickle files
            if save_path:
                if not week_features.isna().values.any():
                    with open(save_path+str(week_date.date())+".pkl", 'wb') as f:
                        pickle.dump(week_features, f)  
                else:
                    print(week_date.date())   

            # Save predicted volatility
            week_features['target_vol'] = target_vol
            predicted_vols[week_date] = week_features['target_vol']


        # Combine all weeks
        predicted_vols = pd.DataFrame(predicted_vols).T
        features_df = pd.concat(all_features, names=['date', 'ticker'])
        
        print(f"\n✓ Weekly feature computation complete!")
        print(f"  Shape: {features_df.shape}")
        print(f"  Features: {features_df.shape[1]}")
        print(f"  Weeks processed: {len(all_features)}")
        
        return features_df, predicted_vols


    def _get_weekly_dates(self):
        """
        Get weekly rebalance dates (Mondays or first trading day of week)
        
        Returns:
            List of weekly dates
        """
        # Resample to weekly (Monday)
        weekly_dates = self.returns.resample('W-MON').first().dropna().index.tolist()
        
        # Filter to only dates that exist in our data
        weekly_dates = [d for d in weekly_dates if d in self.dates]
        
        # Ensure we have enough lookback history
        min_date = self.dates[self.lookback]
        weekly_dates = [d for d in weekly_dates if d >= min_date]
        
        return weekly_dates


    def _aggregate_to_weekly(self, 
                            daily_df: pd.DataFrame, 
                            method: str = 'sum') -> pd.DataFrame:
        """
        Aggregate daily data to weekly
        
        Args:
            daily_df: Daily data (dates × stocks)
            method: Aggregation method
                    'sum' - for returns, volume
                    'last' - for close price
                    'max' - for high
                    'min' - for low
                    'mean' - for averages
        
        Returns:
            Weekly aggregated data
        """
        if method == 'sum':
            weekly = daily_df.resample('W-MON').sum()
        elif method == 'last':
            weekly = daily_df.resample('W-MON').last()
        elif method == 'max':
            weekly = daily_df.resample('W-MON').max()
        elif method == 'min':
            weekly = daily_df.resample('W-MON').min()
        elif method == 'mean':
            weekly = daily_df.resample('W-MON').mean()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        return weekly

    def _calculate_target_volatility(self, returns_next_week: pd.DataFrame) -> pd.Series:
        """
        Calculate realized variance for next week
        
        RV = sum of squared daily returns
        
        Args:
            returns_next_week: Daily returns for next week (5 days × stocks)
        
        Returns:
            Series with realized variance per stock (annualized)
        """
        n_stocks = returns_next_week.shape[1]
        realized_variances = np.zeros(n_stocks)
        
        for i in range(n_stocks):
            daily_returns = returns_next_week.iloc[:, i].dropna()
            
            if len(daily_returns) > 0:
                # Sum of squared daily returns
                rv = np.sum(daily_returns ** 2)
                
                # Annualize: RV is for 5 days, annualize to 252 days
                # RV_annual = RV_5day × (252/5)
                rv_annualized = rv * (252 / len(daily_returns))
                
                realized_variances[i] = rv_annualized
            else:
                realized_variances[i] = np.nan
        
        return pd.Series(realized_variances, index=returns_next_week.columns)