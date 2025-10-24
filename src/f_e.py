from .imports import *

class FeatureEngineer:
    """
    Compute time-series and cross-sectional features from OHLC data
    """
    
    def __init__(self, 
                 open: pd.DataFrame,
                 high: pd.DataFrame,
                 low: pd.DataFrame,
                 close: pd.DataFrame,
                 volume: pd.DataFrame,
                 returns: pd.DataFrame,
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

    def _compute_features_and_targets_generic(self,rebalance_dates, horizon_days: int, target: str, save_path: str = None) -> pd.DataFrame:
        """
        Generic method for any frequency/horizon combination
        
        Args:
            rebalance_dates: List of dates to compute features for
            horizon_days: Prediction horizon in days
            target: Target variable to compute ('return', 'volatility')
            save_path: Optional path to save features as pickle
        
        Returns:
            all_features: DataFrame with computed features
            targets_df: DataFrame with target variable
        """
        print("\nComputing features...")
        
        # Only compute for dates with sufficient history
        start_date = self.dates[self.lookback]
        valid_dates = [d for d in rebalance_dates if d >= start_date]
        
        all_features = {}
        targets_dict = {}

        print(f"  Total dates to compute: {len(valid_dates)}")
        
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
        
            # Compute target
            if target == 'return':
                future_returns = self.returns.iloc[date_idx : date_idx+1+horizon_days]
                targets = future_returns.sum(axis=0)  # Sum across days
            else:
                future_returns = self.returns.iloc[date_idx+1 : date_idx+1+horizon_days]
                rv = (future_returns ** 2).sum(axis=0)
                targets = rv * (252 / horizon_days)  # Annualise 
            targets_dict[date] = targets
        
        targets_dict = pd.DataFrame(targets_dict).T

        if save_path:
            if not targets_dict.isna().values.any():
                with open(f"{save_path}{target}_{horizon_days}_targets.pkl", 'wb') as f:
                    pickle.dump(targets_dict, f)  
            else:
                print(date) 


        # Combine all dates
        features_df = pd.concat(all_features, ignore_index=True)
        
        print(f"\n✓ Feature computation complete!")
        print(f"  Shape: {features_df.shape}")
        print(f"  Features: {features_df.shape[1]}")
        print(f"  Memory: {features_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        
        return features_df, targets_dict
    
    def compute_daily_return_features(self, save_path=None):
        dates = self.dates[self.lookback:]  # All dates
        return self._compute_features_and_targets_generic(
            dates, horizon_days=1, target='return', save_path=save_path
        )
    
    def compute_weekly_return_features(self, save_path=None):
        dates = self._get_weekly_dates()
        return self._compute_features_and_targets_generic(
            dates, horizon_days=5, target='return', save_path=save_path
        )
    
    def compute_monthly_return_features(self, save_path=None):
        dates = self._get_monthly_dates()
        return self._compute_features_and_targets_generic(
            dates, horizon_days=21, target='return', save_path=save_path
        )
    
    def compute_weekly_volatility_features(self, save_path=None):
        dates = self._get_weekly_dates()
        return self._compute_features_and_targets_generic(
            dates, horizon_days=5, target='volatility', save_path=save_path
        )
    
    def compute_monthly_volatility_features(self, save_path=None):
        dates = self._get_monthly_dates()
        return self._compute_features_and_targets_generic(
            dates, horizon_days=21, target='volatility', save_path=save_path
        )

    def _get_weekly_dates(self):
        """
        Get weekly rebalance dates (Mondays or first trading day of week)
        
        Returns:
            List of weekly dates
        """
        # Resample to weekly (Monday)
        weekly_dates = self.returns.groupby([self.returns.index.year, self.returns.index.isocalendar().week]).apply(lambda x: x.index[0]).tolist()
        # Ensure we have enough lookback history
        min_date = self.dates[self.lookback]
        weekly_dates = [d for d in weekly_dates if d >= min_date]
        
        return weekly_dates
    
    def _get_monthly_dates(self):
        """First trading day of each month"""
        monthly_dates = self.returns.groupby([self.returns.index.year, self.returns.index.month]).apply(lambda x: x.index[0]).tolist()
        min_date = self.dates[self.lookback]
        return [d for d in monthly_dates if d >= min_date]
    
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
                
                # EWMA volatility
                if len(ret) >= 21:
                    lambda_decay = 0.94
                    ewma_var = 0
                    for i in range(len(ret)-1, max(len(ret)-22, 0), -1):
                        ewma_var = lambda_decay * ewma_var + (1 - lambda_decay) * ret[i]**2
                    feat['vol_ewma'] = np.sqrt(ewma_var) if ewma_var > 0 else 0
                else:
                    feat['vol_ewma'] = 0
                
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