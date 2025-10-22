from .imports import *

class PortfolioOptimizer:
    """
    Portfolio weight optimization methods
    """
    
    def __init__(self, returns: pd.DataFrame):
        """
        Args:
            returns: Historical returns for covariance estimation
        """
        self.returns = returns
        self.n_stocks = returns.shape[1]
        
        # Estimate covariance matrix with shrinkage
        lw = LedoitWolf()
        self.cov_matrix = lw.fit(returns.values).covariance_

    def estimate_covariance(self, returns):
        """
        Estimate covariance matrix using Ledoit-Wolf shrinkage
        
        Args:
            returns: Historical returns (n_samples, n_stocks)
        
        Returns:
            Covariance matrix (n_stocks, n_stocks)
        """
        lw = LedoitWolf()
        cov_matrix = lw.fit(returns).covariance_
        return cov_matrix
        
    def equal_weight(self):
        """
        Equal-weighted portfolio: 1/N for each stock
        """
        return np.ones(self.n_stocks) / self.n_stocks
    
    def minimum_variance(self, max_position: float = 0.05):
        """
        Minimum variance portfolio
        
        Args:
            max_position: Maximum weight per stock (e.g., 0.05 = 5%)
        
        Returns:
            Optimal weights (n_stocks,)
        """
        self.cov_matrix = self.estimate_covariance(self.returns)
        def objective(w):
            return 0.5 * w @ self.cov_matrix @ w
        
        # Constraints: weights sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds: long-only, max position size
        bounds = [(0, max_position) for _ in range(self.n_stocks)]
        
        # Initial guess: equal weight
        w0 = np.ones(self.n_stocks) / self.n_stocks
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            print(f"Warning: Optimization did not converge. Using equal weights.")
            return w0
        
        return result.x
    
    def mean_variance(self, predicted_returns: np.ndarray, 
                     risk_aversion: float = 1.0,
                     max_position: float = 0.05) -> np.ndarray:
        """
        Mean-variance optimization
        
        max: w^T * mu - (lambda/2) * w^T * Sigma * w
        
        Args:
            predicted_returns: Expected returns (n_stocks,)
            risk_aversion: Risk aversion parameter (higher = more conservative)
            max_position: Maximum weight per stock
        
        Returns:
            Optimal weights (n_stocks,)
        """
        self.cov_matrix = self.estimate_covariance(self.returns)
        def objective(w):
            portfolio_return = w @ predicted_returns
            portfolio_variance = w @ self.cov_matrix @ w
            return -(portfolio_return - (risk_aversion / 2) * portfolio_variance)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        bounds = [(0, max_position) for _ in range(self.n_stocks)]
        
        w0 = np.ones(self.n_stocks) / self.n_stocks
        
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            print(f"Warning: Optimization did not converge. Using equal weights.")
            return w0
        
        return result.x
    
    def target_return_maximization(self,
                                 target_return: float,
                                 max_position: float = 0.05) -> np.ndarray:
        """
        For a given target return, minimize portfolio variance
        
        Args:
            target_return: Target expected return
            max_position: Maximum weight per stock

        Returns:
            Optimal weights (n_stocks,)
        """
        self.cov_matrix = self.estimate_covariance(self.returns)
        returns = self.returns.iloc[-1].values  # Use most recent returns as expected returns
        def objective(w):
            portfolio_variance = w @ self.cov_matrix @ w
            return portfolio_variance

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # weights sum to 1
            {'type': 'ineq', 'fun': lambda w: float(np.dot(w, returns) - target_return)}
        ]

        bounds = [(0, max_position) for _ in range(self.n_stocks)]

        w0 = np.ones(self.n_stocks) / self.n_stocks

        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        if not result.success:
            print(f"Warning: Optimization did not converge. Using equal weights.")
            return w0

        return result.x

    def load_pretrained_weights(self, filepath: str) -> np.ndarray:
        """
        Load pre-trained weights from a file
        
        Args:
            filepath: Path to the file containing weights (e.g., .npy file)
        
        Returns:
            Weights array (n_stocks,)
        """
        weights = np.load(filepath)
        if weights.shape[0] != self.n_stocks:
            raise ValueError(f"Loaded weights shape {weights.shape} does not match number of stocks {self.n_stocks}.")
        return weights / np.sum(weights)  # Normalize to sum to 1