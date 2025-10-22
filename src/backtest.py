from .imports import *

class WalkForwardBacktest:
    """
    Walk-forward backtesting framework
    """
    
    def __init__(self, 
                 train_returns: pd.DataFrame,
                 test_returns: pd.DataFrame,
                #  test_prices: pd.DataFrame,
                 rf_rate_test: pd.Series,
                 rebalance_freq: str = 'W'):
        """
        Args:
            train_returns: Training period returns
            test_returns: Test period returns
            test_prices: Test period prices (for rebalancing dates)
            rf_rate_test: Risk-free rate for test period
            rebalance_freq: 'D' (daily), 'W' (weekly), 'M' (monthly)
        """
        self.train_returns = train_returns
        self.test_returns = test_returns
        # self.test_prices = test_prices
        self.rf_rate_test = rf_rate_test
        self.rebalance_freq = rebalance_freq
        
        # Get rebalancing dates
        self.rebalance_dates = self._get_rebalance_dates()
        
    def _get_rebalance_dates(self):
        """
        Get dates when portfolio should be rebalanced
        """
        if self.rebalance_freq == 'D':
            # Every day
            return self.test_returns.index.tolist()
        elif self.rebalance_freq == 'W':
            # Every week (Mondays or first trading day of week)
            return self.test_returns.resample('W-MON').first().dropna().index.tolist()
        elif self.rebalance_freq == 'M':
            # Every month (first trading day)
            return self.test_returns.resample('MS').first().dropna().index.tolist()
        else:
            raise ValueError(f"Unknown rebalance frequency: {self.rebalance_freq}")
    
    def run_backtest(self,
                    strategy_name: str,
                    get_weights_func,
                    transaction_cost_bps: float = 10.0):
        """
        Run walk-forward backtest for a strategy
        
        Args:
            strategy_name: Name of the strategy
            get_weights_func: Function that returns portfolio weights
                             Signature: get_weights_func(train_returns) -> np.ndarray
            transaction_cost_bps: Transaction cost in basis points (10 = 0.1%)
        
        Returns:
            Dictionary with backtest results
        """
        print(f"\nRunning backtest: {strategy_name}")
        print(f"Rebalancing frequency: {self.rebalance_freq}")
        print(f"Rebalancing dates: {len(self.rebalance_dates)}")
        
        # Initialize tracking
        portfolio_values = [1.0]  # Start with $1
        portfolio_weights_history = []
        turnover_history = []
        dates_history = [self.test_returns.index[0]]
        
        # Current weights (start with equal weight or first rebalance weights)
        current_weights = None
        
        # Iterate through test period
        for i, date in enumerate(self.test_returns.index):
            
            # Check if we should rebalance
            if date in self.rebalance_dates:
                # Get new target weights
                target_weights = get_weights_func(self.train_returns.loc[:date])
                # target_weights = get_weights_func(pd.concat([self.train_returns, self.test_returns.iloc[:i]]))
                
                # Calculate turnover if not first rebalance
                if current_weights is not None:
                    # Weights after market movement (before rebalancing)
                    # If yesterday's weights were w, and returns were r,
                    # today's pre-rebalance weights are w * (1+r) / sum(w * (1+r))
                    turnover = np.abs(target_weights-current_weights).sum()
                    
                    # Apply transaction costs
                    transaction_cost = turnover * (transaction_cost_bps / 10000)
                    portfolio_values[-1] *= (1 - transaction_cost)
                else:
                    turnover = 0
                    transaction_cost = 0
                
                turnover_history.append(turnover)
                current_weights = target_weights
                portfolio_weights_history.append((date, target_weights.copy()))
            
            # Calculate portfolio return for today
            if current_weights is not None:
                daily_returns = self.test_returns.iloc[i].values
                portfolio_return = np.dot(current_weights, daily_returns)
                
                # Update portfolio value
                new_value = portfolio_values[-1] * (1 + portfolio_return)
                portfolio_values.append(new_value)
                dates_history.append(date)
                
                # Update weights based on market movement (drift)
                current_weights = current_weights * (1 + daily_returns)
                current_weights = current_weights / current_weights.sum()
            print(i)
        
        # Calculate results
        portfolio_series = pd.Series(portfolio_values[1:], index=dates_history[1:])
        
        results = {
            'strategy_name': strategy_name,
            'portfolio_values': portfolio_series,
            'weights_history': portfolio_weights_history,
            'turnover_history': turnover_history,
            'rebalance_dates': self.rebalance_dates,
            'transaction_cost_bps': transaction_cost_bps
        }
        
        return results

    def run_backtest_preloaded_weights(self,
                                   strategy_name: str,
                                   preloaded_weights: pd.DataFrame,
                                   transaction_cost_bps: float = 10.0):
        """
        Run walk-forward backtest with preloaded weights
        
        Args:
            strategy_name: Name of the strategy
            preloaded_weights: DataFrame (dates × stocks) with portfolio weights
                            Index should be rebalancing dates
                            Each row sums to 1.0
            transaction_cost_bps: Transaction cost in basis points (10 = 0.1%)
        
        Returns:
            Dictionary with backtest results
        """
        print(f"\nRunning backtest: {strategy_name}")
        print(f"Rebalancing frequency: {self.rebalance_freq}")
        print(f"Preloaded weights shape: {preloaded_weights.shape}")
        
        # Initialize tracking
        portfolio_values = []
        portfolio_weights_history = []
        turnover_history = []
        
        current_weights = None
        portfolio_value = 1.0
        
        # Iterate through test period
        for i, date in enumerate(self.test_returns.index):
            
            # Check if we should rebalance
            if date in preloaded_weights.index:
                
                # # Get preloaded weights for this date
                # if date not in preloaded_weights.index:
                #     print(f"Warning: No weights for {date}, skipping rebalance")
                #     continue
                
                target_weights = preloaded_weights.loc[date].values
                         
                # Calculate turnover if not first rebalance
                if current_weights is not None:
                    
                    # Turnover
                    turnover = np.abs(target_weights-current_weights).sum()
                    
                    # Apply transaction costs
                    transaction_cost = turnover * (transaction_cost_bps / 10000)
                    portfolio_value *= (1 - transaction_cost)
                    
                    turnover_history.append(turnover)


                # Update to target weights
                current_weights = target_weights.copy()
                portfolio_weights_history.append((date, target_weights.copy()))
            
            # Apply today's returns
            if current_weights is not None:
                daily_returns = self.test_returns.iloc[i].values
                portfolio_return = np.dot(current_weights, daily_returns)
                
                # Update portfolio value
                portfolio_value *= (1 + portfolio_return)
                
                # Update weights (drift)
                current_weights = current_weights * (1 + daily_returns)
                current_weights = current_weights / current_weights.sum()
            
            portfolio_values.append(portfolio_value)
        
        # Convert to series
        portfolio_series = pd.Series(portfolio_values, index=self.test_returns.index)
        
        print(f"Backtest complete. Final value: ${portfolio_series.iloc[-1]:.4f}")

        results = {
            'strategy_name': strategy_name,
            'portfolio_values': portfolio_series,
            'weights_history': portfolio_weights_history,
            'turnover_history': turnover_history,
            'rebalance_dates': self.rebalance_dates,
            'transaction_cost_bps': transaction_cost_bps
        }
        
        return results