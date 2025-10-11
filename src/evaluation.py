from .imports import *

class PerformanceEvaluator:
    """
    Calculate and compare performance metrics
    """
    
    def __init__(self, rf_rate_daily: pd.Series):
        """
        Args:
            rf_rate_daily: Daily risk-free rate
        """
        self.rf_rate_daily = rf_rate_daily
    
    def calculate_metrics(self, portfolio_series: pd.Series, 
                         turnover_history,
                         freq: str = 'D'):
        """
        Calculate comprehensive performance metrics
        
        Args:
            portfolio_series: Time series of portfolio values
            turnover_history: List of turnover values at each rebalance
            freq: Frequency of returns ('D', 'W', 'M')
        
        Returns:
            Dictionary of metrics
        """
        # Annualization factor
        if freq == 'D':
            ann_factor = 252
        elif freq == 'W':
            ann_factor = 52
        elif freq == 'M':
            ann_factor = 12
        else:
            ann_factor = 252
        
        # Calculate returns
        returns = portfolio_series.pct_change().dropna()
        
        # Align risk-free rate
        rf_aligned = self.rf_rate_daily.reindex(returns.index, method='ffill')
        if isinstance(rf_aligned, pd.DataFrame):
            rf_aligned = rf_aligned.iloc[:, 0]
        
        excess_returns = returns - rf_aligned
        
        # Cumulative return
        total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1
        
        # Annualized return
        n_years = len(portfolio_series) / ann_factor
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
        
        # Annualized volatility
        annualized_vol = returns.std() * np.sqrt(ann_factor)
        
        # Sharpe ratio
        sharpe = np.sqrt(ann_factor) * excess_returns.mean() / excess_returns.std()
        
        # Sortino ratio (downside deviation)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.sqrt((downside_returns ** 2).mean())
            sortino = np.sqrt(ann_factor) * excess_returns.mean() / downside_std
        else:
            sortino = np.nan
        
        # Maximum drawdown
        cumulative = portfolio_series / portfolio_series.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
        
        # Turnover statistics
        if turnover_history:
            avg_turnover = np.mean(turnover_history)
            total_turnover = np.sum(turnover_history)
        else:
            avg_turnover = 0
            total_turnover = 0
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        metrics = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_vol,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Maximum Drawdown': max_drawdown,
            'Calmar Ratio': calmar,
            'Win Rate': win_rate,
            'Avg Turnover per Rebalance': avg_turnover,
            'Total Turnover': total_turnover
        }
        
        return metrics
    
    def compare_strategies(self, results_dict) -> pd.DataFrame:
        """
        Compare multiple strategies
        
        Args:
            results_dict: Dictionary of {strategy_name: backtest_results}
        
        Returns:
            DataFrame comparing all strategies
        """
        comparison = {}
        
        for name, results in results_dict.items():
            metrics = self.calculate_metrics(
                results['portfolio_values'],
                results.get('turnover_history'),
                freq='D'  # Assumes daily data
            )
            comparison[name] = metrics
        
        df = pd.DataFrame(comparison).T
        
        # Format for display
        df['Total Return'] = df['Total Return'].map('{:.2%}'.format)
        df['Annualized Return'] = df['Annualized Return'].map('{:.2%}'.format)
        df['Annualized Volatility'] = df['Annualized Volatility'].map('{:.2%}'.format)
        df['Sharpe Ratio'] = df['Sharpe Ratio'].map('{:.3f}'.format)
        df['Sortino Ratio'] = df['Sortino Ratio'].map('{:.3f}'.format)
        df['Maximum Drawdown'] = df['Maximum Drawdown'].map('{:.2%}'.format)
        df['Calmar Ratio'] = df['Calmar Ratio'].map('{:.3f}'.format)
        df['Win Rate'] = df['Win Rate'].map('{:.2%}'.format)
        df['Avg Turnover per Rebalance'] = df['Avg Turnover per Rebalance'].map('{:.2%}'.format)
        df['Total Turnover'] = df['Total Turnover'].map('{:.2f}'.format)
        
        return df
    
    def plot_results(self, results_dict, figsize=(15, 10)):
        """
        Create visualization of results
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Cumulative returns
        ax1 = axes[0, 0]
        for name, results in results_dict.items():
            portfolio = results['portfolio_values']
            cumulative = portfolio / portfolio.iloc[0]
            ax1.plot(cumulative.index, cumulative.values, label=name, linewidth=2)
        ax1.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($1 initial)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Drawdowns
        ax2 = axes[0, 1]
        for name, results in results_dict.items():
            portfolio = results['portfolio_values']
            cumulative = portfolio / portfolio.iloc[0]
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            ax2.plot(drawdown.index, drawdown.values, label=name, linewidth=2)
        ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 3: Rolling Sharpe Ratio (60-day window)
        ax3 = axes[1, 0]
        for name, results in results_dict.items():
            portfolio = results['portfolio_values']
            returns = portfolio.pct_change().dropna()
            rf_aligned = self.rf_rate_daily.reindex(returns.index, method='ffill')
            if isinstance(rf_aligned, pd.DataFrame):
                rf_aligned = rf_aligned.iloc[:, 0]
            excess_returns = returns - rf_aligned
            
            rolling_sharpe = (excess_returns.rolling(60).mean() / 
                            excess_returns.rolling(60).std() * np.sqrt(252))
            ax3.plot(rolling_sharpe.index, rolling_sharpe.values, label=name, linewidth=2)
        ax3.set_title('Rolling 60-Day Sharpe Ratio', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 4: Distribution of returns
        ax4 = axes[1, 1]
        for name, results in results_dict.items():
            portfolio = results['portfolio_values']
            returns = portfolio.pct_change().dropna()
            ax4.hist(returns, bins=50, alpha=0.5, label=name, density=True)
        ax4.set_title('Distribution of Daily Returns', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Daily Return')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig