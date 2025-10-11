from .imports import *

class ReturnPredictor:
    """
    Return prediction methods (starting with simple baselines)
    """
    
    def __init__(self, train_returns):
        """
        Args:
            train_returns: Historical returns for training
        """
        self.train_returns = train_returns
        self.n_stocks = train_returns.shape[1]
        
    def predict_historical_mean(self, lookback_days = None):
        """
        Predict using historical mean return
        
        Args:
            lookback_days: If None, use all training data. Otherwise, use last N days.
        
        Returns:
            Array of predicted returns (n_stocks,)
        """
        if lookback_days is None:
            returns_to_use = self.train_returns
        else:
            returns_to_use = self.train_returns.iloc[-lookback_days:]
        
        raw_mean = returns_to_use.mean().values
        
        shrinkage_factor = 1
        predicted_returns = shrinkage_factor * raw_mean
        
        return predicted_returns
    
    def predict_equal_expected(self):
        """
        Predict equal expected returns for all stocks (zero alpha assumption)
        
        Returns:
            Array of zeros (n_stocks,)
        """
        return np.zeros(self.n_stocks)