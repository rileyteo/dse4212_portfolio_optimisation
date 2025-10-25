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

            date_features = self.feature_subset_horizon(date_features, horizon_days, target)   

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
    
    # def _compute_features_single_date(self,
    #                                  returns_window: pd.DataFrame,
    #                                  excess_returns_window: pd.DataFrame,
    #                                  close_window: pd.DataFrame,
    #                                  volume_window: pd.DataFrame,
    #                                  high_window: pd.DataFrame,
    #                                  low_window: pd.DataFrame,
    #                                  for_return: bool = True) -> pd.DataFrame:
    #     """
    #     Compute all features for one date across all stocks
    #     """
    #     n_stocks = len(self.tickers)
    #     features_list = []
        
    #     # === TIME-SERIES FEATURES (per stock) ===
    #     for i, ticker in enumerate(self.tickers):
    #         feat = {'ticker': ticker}
            
    #         ret = returns_window.iloc[:, i].values
    #         excess_ret = excess_returns_window.iloc[:, i].values
    #         price = close_window.iloc[:, i].values
    #         vol = volume_window.iloc[:, i].values
    #         high = high_window.iloc[:, i].values
    #         low = low_window.iloc[:, i].values
            
    #         if for_return:
    #             # 1. Momentum (various periods)
    #             feat['ret_1d'] = ret[-1] if len(ret) > 0 else 0
    #             feat['ret_5d'] = np.sum(ret[-5:]) if len(ret) >= 5 else 0
    #             feat['ret_21d'] = np.sum(ret[-21:]) if len(ret) >= 21 else 0
    #             feat['ret_63d'] = np.sum(ret[-63:]) if len(ret) >= 63 else 0
                
    #             # 2. Excess returns (risk-adjusted momentum)
    #             feat['excess_ret_21d'] = np.sum(excess_ret[-21:]) if len(excess_ret) >= 21 else 0
    #             feat['excess_ret_63d'] = np.sum(excess_ret[-63:]) if len(excess_ret) >= 63 else 0
                
    #             # 3. Volatility
    #             feat['vol_5d'] = np.std(ret[-5:]) if len(ret) >= 5 else 0
    #             feat['vol_21d'] = np.std(ret[-21:]) if len(ret) >= 21 else 0
    #             feat['vol_63d'] = np.std(ret[-63:]) if len(ret) >= 63 else 0
                
    #             # 4. Sharpe-like ratio (excess return / volatility)
    #             if len(excess_ret) >= 21 and feat['vol_21d'] > 0:
    #                 feat['sharpe_21d'] = feat['excess_ret_21d'] / (feat['vol_21d'] * np.sqrt(21))
    #             else:
    #                 feat['sharpe_21d'] = 0
                
    #             # 5. Moving average ratios
    #             if len(price) >= 21:
    #                 ma20 = np.mean(price[-21:])
    #                 feat['price_ma20'] = (price[-1] / ma20) - 1
    #             else:
    #                 feat['price_ma20'] = 0
                
    #             if len(price) >= 50:
    #                 ma50 = np.mean(price[-50:])
    #                 feat['price_ma50'] = (price[-1] / ma50) - 1
    #             else:
    #                 feat['price_ma50'] = 0
                
    #             # 6. Momentum acceleration (recent vs older)
    #             if len(ret) >= 21:
    #                 recent = np.sum(ret[-5:])
    #                 older = np.sum(ret[-21:-5]) / 16 * 5  # Normalize to same period
    #                 feat['momentum_accel'] = recent - older
    #             else:
    #                 feat['momentum_accel'] = 0
                
    #             # 7. Volume ratio
    #             if len(vol) >= 21:
    #                 avg_vol = np.mean(vol[-21:])
    #                 current_vol = vol[-1]
    #                 feat['volume_ratio'] = current_vol / avg_vol if avg_vol > 0 else 1
    #             else:
    #                 feat['volume_ratio'] = 1
                
    #             # 8. High-low range position
    #             if len(high) >= 21:
    #                 high_21 = np.max(high[-21:])
    #                 low_21 = np.min(low[-21:])
    #                 if high_21 != low_21:
    #                     feat['range_position'] = (price[-1] - low_21) / (high_21 - low_21)
    #                 else:
    #                     feat['range_position'] = 0.5
    #             else:
    #                 feat['range_position'] = 0.5
                
    #             # 9. Drawdown from peak
    #             if len(price) >= 21:
    #                 peak = np.max(price[-21:])
    #                 feat['drawdown_21d'] = (price[-1] / peak) - 1
    #             else:
    #                 feat['drawdown_21d'] = 0
                
    #             # 10. Skewness and kurtosis
    #             if len(ret) >= 21:
    #                 feat['skew_21d'] = stats.skew(ret[-21:])
    #                 feat['kurt_21d'] = stats.kurtosis(ret[-21:])
    #             else:
    #                 feat['skew_21d'] = 0
    #                 feat['kurt_21d'] = 0
                
    #             # 11. RSI 14
    #             if len(price) >= 15:
    #                 deltas = np.diff(price[-15:])
    #                 ups = deltas[deltas > 0].sum() / 14
    #                 downs = -deltas[deltas < 0].sum() / 14
    #                 rs = ups / downs if downs > 0 else 0
    #                 feat['rsi_14'] = 100 - (100 / (1 + rs))

    #             # 12. Lagged returns
    #             for i in range(1, 21):
    #                 feat[f'ret_lag_{i}d'] = ret[-i] if len(ret) >= i else 0
                
    #             # EWMA volatility
    #             if len(ret) >= 21:
    #                 lambda_decay = 0.94
    #                 ewma_var = 0
    #                 for i in range(len(ret)-1, max(len(ret)-22, 0), -1):
    #                     ewma_var = lambda_decay * ewma_var + (1 - lambda_decay) * ret[i]**2
    #                 feat['vol_ewma'] = np.sqrt(ewma_var) if ewma_var > 0 else 0
    #             else:
    #                 feat['vol_ewma'] = 0
                
    #         else: # For volatility features
    #             # 1. Volatility
    #             feat['vol_5d'] = np.std(ret[-5:]) if len(ret) >= 5 else 0
    #             feat['vol_21d'] = np.std(ret[-21:]) if len(ret) >= 21 else 0
    #             feat['vol_63d'] = np.std(ret[-63:]) if len(ret) >= 63 else 0 

    #             # 2. Volume ratio
    #             if len(vol) >= 21:
    #                 avg_vol = np.mean(vol[-21:])
    #                 current_vol = vol[-1]
    #                 feat['volume_ratio'] = current_vol / avg_vol if avg_vol > 0 else 1
    #             else:
    #                 feat['volume_ratio'] = 1                   

    #             # 4. Skewness and kurtosis
    #             if len(ret) >= 21:
    #                 feat['skew_21d'] = stats.skew(ret[-21:])
    #                 feat['kurt_21d'] = stats.kurtosis(ret[-21:])
    #             else:
    #                 feat['skew_21d'] = 0
    #                 feat['kurt_21d'] = 0

    #             # 5 . Recent Shocks
    #             feat['rv_5d'] = np.mean(ret[-5:]**2) if len(ret) >= 5 else 0
    #             feat['rv_21d'] = np.mean(ret[-21:]**2) if len(ret) >= 21 else 0
    #             feat['rv_63d'] = np.mean(ret[-63:]**2) if len(ret) >= 63 else 0

    #             #6. Realised variance ratios
    #             feat['rv_ratio_5_21'] = feat['rv_5d'] / feat['rv_21d'] if feat['rv_21d'] > 0 else 0
    #             feat['rv_ratio_5_63'] = feat['rv_5d'] / feat['rv_63d'] if feat['rv_63d'] > 0 else 0
    #             feat['rv_ratio_21_63'] = feat['rv_21d'] / feat['rv_63d'] if feat['rv_63d'] > 0 else 0

    #             #7. Realised variance trend
    #             feat['rv_trend'] = (feat['rv_21d'] - feat['rv_63d']) / feat['rv_63d'] if feat['rv_63d'] > 0 else 0

    #             # 8. EWMA volatility
    #             if len(ret) >= 21:
    #                 lambda_decay = 0.94
    #                 ewma_var = 0
    #                 for i in range(len(ret)-1, max(len(ret)-22, 0), -1):
    #                     ewma_var = lambda_decay * ewma_var + (1 - lambda_decay) * ret[i]**2
    #                 feat['vol_ewma'] = np.sqrt(ewma_var) if ewma_var > 0 else 0
    #             else:
    #                 feat['vol_ewma'] = 0

    #             # 9. Volume Ratio
    #             if len(vol) >= 21:
    #                 avg_vol = np.mean(vol[-21:])
    #                 current_vol = vol[-1]
    #                 feat['volume_ratio'] = current_vol / avg_vol if avg_vol > 0 else 1
    #             else:
    #                 feat['volume_ratio'] = 1


    #         features_list.append(feat)
            

        
    #     df = pd.DataFrame(features_list)
        
    #     # === CROSS-SECTIONAL FEATURES (compare stocks) ===
        
    #     # Market (equal-weighted)
    #     market_ret = returns_window.mean(axis=1)
        
    #     if for_return:
    #         # 1. Percentile ranks
    #         for col in ['ret_21d', 'ret_63d', 'vol_21d', 'excess_ret_21d']:
    #             df[f'{col}_pct'] = df[col].rank(pct=True)
            
    #         # 2. Relative to median
    #         for col in ['ret_21d', 'ret_63d', 'vol_21d']:
    #             median = df[col].median()
    #             df[f'{col}_vs_med'] = df[col] - median
            
    #         # 3. Z-scores
    #         for col in ['ret_21d', 'vol_21d']:
    #             mean = df[col].mean()
    #             std = df[col].std()
    #             if std > 0:
    #                 df[f'{col}_zscore'] = (df[col] - mean) / std
    #             else:
    #                 df[f'{col}_zscore'] = 0
            
    #         # 4. Beta to market
    #         betas = []
    #         for i in range(n_stocks):
    #             stock_ret = returns_window.iloc[:, i].values
    #             if len(stock_ret) >= 60 and len(market_ret) >= 60:
    #                 cov = np.cov(stock_ret[-60:], market_ret[-60:].values)[0, 1]
    #                 var = np.var(market_ret[-60:].values)
    #                 beta = cov / var if var > 0 else 1
    #             else:
    #                 beta = 1
    #             betas.append(beta)
            
    #         df['beta'] = betas
    #         df['beta_pct'] = pd.Series(betas).rank(pct=True).values
            
    #         # 5. Correlation to market
    #         correlations = []
    #         for i in range(n_stocks):
    #             stock_ret = returns_window.iloc[:, i].values
    #             if len(stock_ret) >= 60 and len(market_ret) >= 60:
    #                 corr = np.corrcoef(stock_ret[-60:], market_ret[-60:].values)[0, 1]
    #             else:
    #                 corr = 0
    #             correlations.append(corr)
            
    #         df['corr_market'] = correlations
            
    #         # 6. Risk-adjusted momentum (interaction feature)
    #         df['risk_adj_mom'] = df['excess_ret_21d'] / (df['vol_21d'] + 1e-8)
    #         df['risk_adj_mom_pct'] = df['risk_adj_mom'].rank(pct=True)
            
    #         # 7. Momentum-volatility interaction
    #         df['mom_vol_interaction'] = df['ret_21d'] * df['vol_21d']
        
        
    #     else:
    #         # 1. Percentile ranks
    #         for col in ['vol_21d', 'vol_5d', 'vol_63d']:
    #             df[f'{col}_pct'] = df[col].rank(pct=True)
            
    #         # 2. Relative to median
    #         for col in ['vol_21d']:
    #             median = df[col].median()
    #             df[f'{col}_vs_med'] = df[col] - median
            
    #         # 3. Z-scores
    #         for col in ['vol_21d', 'vol_5d', 'vol_63d']:
    #             mean = df[col].mean()
    #             std = df[col].std()
    #             if std > 0:
    #                 df[f'{col}_zscore'] = (df[col] - mean) / std
    #             else:
    #                 df[f'{col}_zscore'] = 0  
    #     return df

    def _compute_features_single_date(self,
                                    returns_window: pd.DataFrame,
                                    excess_returns_window: pd.DataFrame,
                                    close_window: pd.DataFrame,
                                    volume_window: pd.DataFrame,
                                    high_window: pd.DataFrame,
                                    low_window: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ALL features for one date across all stocks
        
        No for_return flag - computes complete feature set
        Feature selection happens in feature_subset_horizon()
        
        Returns:
            DataFrame with ~60-70 features per stock
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
            

            # MOMENTUM FEATURES

            # Short-term momentum
            feat['ret_1d'] = ret[-1] if len(ret) > 0 else 0
            feat['ret_5d'] = np.sum(ret[-5:]) if len(ret) >= 5 else 0
            
            # Medium-term momentum
            feat['ret_21d'] = np.sum(ret[-21:]) if len(ret) >= 21 else 0
            feat['excess_ret_21d'] = np.sum(excess_ret[-21:]) if len(excess_ret) >= 21 else 0
            
            # Long-term momentum
            feat['ret_63d'] = np.sum(ret[-63:]) if len(ret) >= 63 else 0
            feat['excess_ret_63d'] = np.sum(excess_ret[-63:]) if len(excess_ret) >= 63 else 0
            
            # VOLATILITY FEATURES
            
            # Standard deviation measures
            feat['vol_5d'] = np.std(ret[-5:]) if len(ret) >= 5 else 0
            feat['vol_21d'] = np.std(ret[-21:]) if len(ret) >= 21 else 0
            feat['vol_63d'] = np.std(ret[-63:]) if len(ret) >= 63 else 0
            
            # Realised variance (mean of squared returns)
            feat['rv_5d'] = np.mean(ret[-5:]**2) if len(ret) >= 5 else 0
            feat['rv_21d'] = np.mean(ret[-21:]**2) if len(ret) >= 21 else 0
            feat['rv_63d'] = np.mean(ret[-63:]**2) if len(ret) >= 63 else 0
            
            # EWMA volatility
            if len(ret) >= 21:
                lambda_decay = 0.94
                ewma_var = 0
                for j in range(len(ret)-1, max(len(ret)-22, 0), -1):
                    ewma_var = lambda_decay * ewma_var + (1 - lambda_decay) * ret[j]**2
                feat['vol_ewma'] = np.sqrt(ewma_var) if ewma_var > 0 else 0
            else:
                feat['vol_ewma'] = 0
            
            # Downside volatility (semi-deviation)
            if len(ret) >= 21:
                downside_returns = ret[-21:][ret[-21:] < 0]
                feat['downside_vol_21d'] = np.std(downside_returns) if len(downside_returns) > 0 else 0
            else:
                feat['downside_vol_21d'] = 0
            
            # ==========================================
            # VOLATILITY DYNAMICS
            # ==========================================
            
            # Realised variance ratios
            feat['rv_ratio_5_21'] = feat['rv_5d'] / feat['rv_21d'] if feat['rv_21d'] > 0 else 0
            feat['rv_ratio_5_63'] = feat['rv_5d'] / feat['rv_63d'] if feat['rv_63d'] > 0 else 0
            feat['rv_ratio_21_63'] = feat['rv_21d'] / feat['rv_63d'] if feat['rv_63d'] > 0 else 0
            
            # Realised variance trend
            feat['rv_trend'] = (feat['rv_21d'] - feat['rv_63d']) / feat['rv_63d'] if feat['rv_63d'] > 0 else 0
            
            # ==========================================
            # TECHNICAL INDICATORS
            # ==========================================
            
            # Moving average ratios
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
            
            # Momentum acceleration
            if len(ret) >= 21:
                recent = np.sum(ret[-5:])
                older = np.sum(ret[-21:-5]) / 16 * 5  # Normalise to same period
                feat['momentum_accel'] = recent - older
            else:
                feat['momentum_accel'] = 0
            
            # High-low range position
            if len(high) >= 21:
                high_21 = np.max(high[-21:])
                low_21 = np.min(low[-21:])
                if high_21 != low_21:
                    feat['range_position'] = (price[-1] - low_21) / (high_21 - low_21)
                else:
                    feat['range_position'] = 0.5
            else:
                feat['range_position'] = 0.5
            
            # RSI (14-period)
            if len(price) >= 15:
                deltas = np.diff(price[-15:])
                ups = deltas[deltas > 0].sum() / 14
                downs = -deltas[deltas < 0].sum() / 14
                rs = ups / downs if downs > 0 else 0
                feat['rsi_14'] = 100 - (100 / (1 + rs))
            else:
                feat['rsi_14'] = 50
            
            # Drawdown from peak
            if len(price) >= 21:
                peak = np.max(price[-21:])
                feat['drawdown_21d'] = (price[-1] / peak) - 1
            else:
                feat['drawdown_21d'] = 0
            
            # ==========================================
            # LAGGED RETURNS (for daily predictions)
            # ==========================================
            
            for lag in range(1, 21):
                feat[f'ret_lag_{lag}d'] = ret[-lag] if len(ret) >= lag else 0
            
            # ==========================================
            # VOLUME FEATURES
            # ==========================================
            
            # Volume ratio
            if len(vol) >= 21:
                avg_vol = np.mean(vol[-21:])
                current_vol = vol[-1]
                feat['volume_ratio'] = current_vol / avg_vol if avg_vol > 0 else 1
            else:
                feat['volume_ratio'] = 1
            
            # Volume volatility
            if len(vol) >= 21:
                feat['volume_vol_21d'] = np.std(vol[-21:]) / (np.mean(vol[-21:]) + 1e-8)
            else:
                feat['volume_vol_21d'] = 0
            
            # ==========================================
            # LIQUIDITY FEATURES
            # ==========================================
            
            if len(ret) >= 21 and len(vol) >= 21:
                # Amihud illiquidity measure
                abs_returns_sum = np.abs(ret[-21:]).sum()
                volume_sum = vol[-21:].sum()
                feat['illiquidity_21d'] = abs_returns_sum / (volume_sum + 1e-8)
                
                # High-low spread proxy
                hl_spreads = (high[-21:] - low[-21:]) / (close_window.iloc[:, i].values[-21:] + 1e-8)
                feat['hl_spread_21d'] = np.mean(hl_spreads)
            else:
                feat['illiquidity_21d'] = 0
                feat['hl_spread_21d'] = 0
            
            # ==========================================
            # RISK FEATURES
            # ==========================================
            
            # Sharpe-like ratio
            if len(excess_ret) >= 21 and feat['vol_21d'] > 0:
                feat['sharpe_21d'] = feat['excess_ret_21d'] / (feat['vol_21d'] * np.sqrt(21))
            else:
                feat['sharpe_21d'] = 0
            
            # Skewness and kurtosis
            if len(ret) >= 21:
                feat['skew_21d'] = stats.skew(ret[-21:])
                feat['kurt_21d'] = stats.kurtosis(ret[-21:])
            else:
                feat['skew_21d'] = 0
                feat['kurt_21d'] = 0
            
            # ==========================================
            # TAIL/JUMP FEATURES
            # ==========================================
            
            # Maximum absolute return
            if len(ret) >= 21:
                feat['max_ret_21d'] = np.max(np.abs(ret[-21:]))
                feat['min_ret_21d'] = np.min(ret[-21:])
            else:
                feat['max_ret_21d'] = 0
                feat['min_ret_21d'] = 0
            
            # Tail ratio (95th percentile / 5th percentile)
            if len(ret) >= 63:
                p95 = np.percentile(ret[-63:], 95)
                p5 = np.percentile(ret[-63:], 5)
                feat['tail_ratio'] = p95 / abs(p5) if p5 != 0 else 1
            else:
                feat['tail_ratio'] = 1
            
            features_list.append(feat)
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        
        # ==========================================
        # CROSS-SECTIONAL FEATURES
        # ==========================================
        
        # Market (equal-weighted)
        market_ret = returns_window.mean(axis=1)
        
        # Percentile ranks for momentum
        for col in ['ret_21d', 'ret_63d', 'excess_ret_21d']:
            if col in df.columns:
                df[f'{col}_pct'] = df[col].rank(pct=True)
        
        # Percentile ranks for volatility
        for col in ['vol_21d', 'vol_5d', 'vol_63d']:
            if col in df.columns:
                df[f'{col}_pct'] = df[col].rank(pct=True)
        
        # Z-scores
        for col in ['ret_21d', 'vol_21d']:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[f'{col}_zscore'] = (df[col] - mean) / std
                else:
                    df[f'{col}_zscore'] = 0
        
        # Beta to market
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
        
        # Correlation to market
        correlations = []
        for i in range(n_stocks):
            stock_ret = returns_window.iloc[:, i].values
            if len(stock_ret) >= 60 and len(market_ret) >= 60:
                corr = np.corrcoef(stock_ret[-60:], market_ret[-60:].values)[0, 1]
            else:
                corr = 0
            correlations.append(corr)
        
        df['corr_market'] = correlations
        
        # Risk-adjusted momentum
        df['risk_adj_mom'] = df['excess_ret_21d'] / (df['vol_21d'] + 1e-8)
        df['risk_adj_mom_pct'] = df['risk_adj_mom'].rank(pct=True)
        
        # Momentum-volatility interaction
        df['mom_vol_interaction'] = df['ret_21d'] * df['vol_21d']
        
        return df


    def feature_subset_horizon(self, features: pd.DataFrame, horizon_days: int, target: str) -> pd.DataFrame:
        """
        Select feature subsets based on prediction horizon and target type
        
        Uses predefined feature pools for consistency and maintainability
        
        Args:
            features: DataFrame with all features
            horizon_days: Prediction horizon in days (1, 5, or 21)
            target: Target variable ('return' or 'volatility')
        
        Returns:
            DataFrame with selected features
        """
        
        # ==========================================
        # FEATURE POOL DEFINITIONS
        # ==========================================
        
        pools = {
            # Momentum features (different horizons)
            'momentum_short': ['ret_1d', 'ret_5d'],
            'momentum_medium': ['ret_21d', 'excess_ret_21d'],
            'momentum_long': ['ret_63d', 'excess_ret_63d'],
            
            # Volatility measures
            'volatility_short': ['vol_5d', 'rv_5d'],
            'volatility_medium': ['vol_21d', 'rv_21d', 'vol_ewma', 'downside_vol_21d'],
            'volatility_long': ['vol_63d', 'rv_63d'],
            
            # Volatility dynamics
            'vol_dynamics': ['rv_ratio_5_21', 'rv_ratio_5_63', 'rv_ratio_21_63', 'rv_trend'],
            
            # Technical indicators
            'technical_short': [f'ret_lag_{i}d' for i in range(1, 6)],  # Lags 1-5
            'technical_medium': ['price_ma20', 'momentum_accel', 'range_position', 'rsi_14'],
            'technical_long': ['price_ma50', 'drawdown_21d'],
            
            # Volume features
            'volume': ['volume_ratio', 'volume_vol_21d'],
            
            # Risk features
            'risk': ['sharpe_21d', 'skew_21d', 'kurt_21d'],
            
            # Liquidity features
            'liquidity': ['illiquidity_21d', 'hl_spread_21d'],
            
            # Tail/jump features
            'tail': ['max_ret_21d', 'min_ret_21d', 'tail_ratio'],
            
            # Cross-sectional features
            'cross_sectional': [
                'ret_21d_pct', 'ret_63d_pct', 'excess_ret_21d_pct',
                'vol_21d_pct', 'vol_5d_pct', 'vol_63d_pct',
                'ret_21d_zscore', 'vol_21d_zscore',
                'beta', 'beta_pct', 'corr_market',
                'risk_adj_mom', 'risk_adj_mom_pct', 'mom_vol_interaction'
            ]
        }
        
        # ==========================================
        # SELECTION RULES BY (TARGET, HORIZON)
        # ==========================================
        
        selection_rules = {
            # === RETURN PREDICTION ===
            
            ('return', 1): {  # Daily returns
                'pools': [
                    'momentum_short',      # Use very recent returns
                    'volatility_short',    # Recent volatility
                    'volatility_medium',   # For risk adjustment
                    'technical_short',     # Lagged returns 1-5d
                    'technical_medium',    # MA, RSI, momentum accel
                    'volume',              # Volume signals
                    'risk',                # Sharpe, skew, kurtosis
                    'liquidity',           # Illiquidity, spread
                    'cross_sectional'      # Rankings, beta
                ],
                'exclude': [],
                'rationale': 'Daily: Use all high-frequency signals and microstructure'
            },
            
            ('return', 5): {  # Weekly returns
                'pools': [
                    'momentum_short',      # ret_5d still relevant
                    'momentum_medium',     # ret_21d becomes important
                    'volatility_medium',   # 21d vol for risk
                    'technical_medium',    # MA, range position
                    'technical_long',      # Longer MA, drawdown
                    'volume',              # Volume signals
                    'risk',                # Risk metrics
                    'liquidity',           # Liquidity matters
                    'cross_sectional'      # Rankings
                ],
                'exclude': ['ret_1d'] + [f'ret_lag_{i}d' for i in range(1, 6)],  # Drop daily lags
                'rationale': 'Weekly: Drop daily noise, keep medium-term momentum'
            },
            
            ('return', 21): {  # Monthly returns
                'pools': [
                    'momentum_medium',     # ret_21d
                    'momentum_long',       # ret_63d becomes key
                    'volatility_medium',   # 21d vol
                    'volatility_long',     # 63d vol
                    'technical_long',      # Long-term technical
                    'volume',              # Volume
                    'risk',                # Risk metrics
                    'cross_sectional'      # Rankings
                ],
                'exclude': ['vol_5d_pct', 'rv_5d'] +
                        [f'ret_lag_{i}d' for i in range(1, 21)],  # Drop all short-term
                'rationale': 'Monthly: Focus on persistent signals, drop short-term noise'
            },
            
            # === VOLATILITY PREDICTION ===
            
            ('volatility', 5): {  # Weekly volatility
                'pools': [
                    'volatility_short',    # Recent vol
                    'volatility_medium',   # 21d vol, EWMA
                    'vol_dynamics',        # RV ratios, trends
                    'volume',              # Volume signals
                    'risk',                # Skew, kurtosis
                    'tail',                # Jump indicators
                    'liquidity',           # Liquidity affects vol
                    'cross_sectional'      # Rankings
                ],
                'exclude': ['ret_21_d', 'excess_ret_21d', 'ret_63d', 'excess_ret_63d', 'ret_21d_zscore'] ,  # Drop return features
                'rationale': 'Weekly vol: Use recent variance measures and tail risk'
            },
            
            ('volatility', 21): {  # Monthly volatility
                'pools': [
                    'volatility_medium',   # 21d vol
                    'volatility_long',     # 63d vol
                    'vol_dynamics',        # RV dynamics
                    'volume',              # Volume
                    'risk',                # Higher moments
                    'tail',                # Tail risk
                    'liquidity',           # Liquidity
                    'cross_sectional'      # Rankings
                ],
                'exclude': ['vol_5d', 'rv_5d', 'rv_ratio_5_21', 'rv_ratio_5_63'] + ['ret_21_d', 'excess_ret_21d', 'ret_63d', 'excess_ret_63d', 'ret_21d_zscore'],  # Drop short-term
                'rationale': 'Monthly vol: Focus on longer-term variance persistence'
            }
        }
        
        # ==========================================
        # FEATURE SELECTION LOGIC
        # ==========================================
        
        key = (target, horizon_days)
        if key not in selection_rules:
            raise ValueError(f"No selection rule for target={target}, horizon={horizon_days}. "
                            f"Valid combinations: {list(selection_rules.keys())}")
        
        rule = selection_rules[key]
        
        # Build feature list from pools
        selected_features = ['ticker']  # Always include ticker
        for pool_name in rule['pools']:
            if pool_name in pools:
                selected_features.extend(pools[pool_name])
        
        # Remove excluded features
        selected_features = [f for f in selected_features if f not in rule['exclude']]
        
        # Remove duplicates (maintain order)
        seen = set()
        selected_features = [f for f in selected_features if not (f in seen or seen.add(f))]
        
        # Filter to features that actually exist in DataFrame
        available_features = [f for f in selected_features if f in features.columns]
        
        # Warn about missing features
        missing = set(selected_features) - set(available_features) - {'ticker'}
        if missing:
            print(f"⚠ Warning: {len(missing)} expected features not found in data:")
            print(f"  Missing: {sorted(missing)[:10]}")  # Show first 10
        
        # Print summary
        print(f"\n✓ Selected {len(available_features)-1} features for {target} prediction (horizon={horizon_days}d)")
        print(f"  Pools used: {rule['pools']}")
        if rule['exclude']:
            print(f"  Excluded: {len(rule['exclude'])} features")
        
        return features[available_features]