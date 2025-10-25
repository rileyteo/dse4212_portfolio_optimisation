from .imports import *

class MLReturnPredictor:
    """ML-based return prediction"""
    
    def __init__(self, model_type: str = 'ridge', **model_params):
        """
        Args:
            model_type: 'ridge' or 'random_forest' or 'Lasso' or 'ENet' or 'XGB'
            **model_params: Model hyperparameters
        """
        self.model_type = model_type
        self.model_params = model_params
        self.model = None
    
    def create_training_dataset(self,
                               feature_files,
                               returns_df: pd.DataFrame):
        """
        Create training dataset from feature files and returns
        
        Args:
            feature_files: List of feature pickle file paths
            returns_df: DataFrame (dates × stocks) with returns
        
        Returns:
            X_train: (N_samples, N_features)
            y_train: (N_samples,)
        """
        print(f"Creating training dataset from {len(feature_files)} files...")
        
        X_list = []
        y_list = []
        for feature_file in feature_files:
            # Extract date from filename
            feature_date = pd.to_datetime(feature_file.replace(".pkl", ""))
            
            # Load features
            with open("src/data/"+feature_file, 'rb') as f:
                features_df = pickle.load(f)
            
            # Get target returns (1-day ahead)
            if feature_date not in returns_df.index:
                continue
            
            target_returns = returns_df.loc[feature_date]
            
            # Align stocks
            common_stocks = features_df.index.intersection(target_returns.index)
            if len(common_stocks) == 0:
                continue
            
            # Store
            X_list.append(features_df.loc[common_stocks].values)
            y_list.append(target_returns.loc[common_stocks].values)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        
        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)
        
        print(f"✓ Dataset created: X={X_train.shape}, y={y_train.shape}")
        return X_train, y_train
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        print(f"Training {self.model_type} model...")
        
        if self.model_type == 'ridge':
            params = {'alpha': 1.0, **self.model_params}
            self.model = Ridge(**params)
        elif self.model_type == 'random_forest':
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_leaf': 100,
                'n_jobs': -1,
                **self.model_params
            }
            self.model = RandomForestRegressor(**params)
        elif self.model_type == 'Lasso':
            params = {'alpha': 0.1, **self.model_params}
            self.model = Lasso(**params)
        elif self.model_type == 'ENet':
            params = {'alpha': 0.1, 'l1_ratio': 0.5, **self.model_params}
            self.model = ElasticNet(**params) 
        elif self.model_type == 'XGB':
            params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, **self.model_params}
            self.model = XGBRegressor(**params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.fit(X_train, y_train)
        
        # Quick diagnostics
        y_pred = self.model.predict(X_train)
        r2 = r2_score(y_train, y_pred)
        
        print(f"✓ Training R²: {r2:.4f}")
        print(f"  Prediction spread: {y_pred.std():.4%}")

    def predict_all_test_returns(self,
                                 feature_files):
        """
        Predict returns for all test dates
        
        Args:
            feature_files: List of test feature files
        
        Returns:
            predicted_returns_df: DataFrame (dates × stocks)
        """
        print(f"Predicting returns for {len(feature_files)} dates...")
        
        predictions_dict = {}
        
        for feature_file in feature_files:
            # Extract date
            date_str = Path(feature_file).stem.replace('features_', '')
            pred_date = pd.to_datetime(date_str)
            
            # Load and predict
            with open("src/data/"+feature_file, 'rb') as f:
                features_df = pickle.load(f)
            
            predictions = self.model.predict(features_df.values)
            predictions_dict[pred_date] = pd.Series(predictions, index=features_df.index)
        
        predicted_returns_df = pd.DataFrame(predictions_dict).T
        
        print(f"✓ Predictions complete: {predicted_returns_df.shape}")
        return predicted_returns_df
    
    def save_model(self, path: str):
        """Save model to disk"""
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'model_type': self.model_type}, f)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.model_type = data['model_type']
        print(f"✓ Model loaded from {path}")

# Helper functions
def get_feature_files(feature_dir: str):
    """Get sorted list of feature files"""
    files = glob.glob(str(Path(feature_dir) / 'features_*.pkl'))
    return sorted(files)


def split_feature_files(feature_files, 
                       train_years: int = 4):
    """Split into train/test sets"""
    split_idx = min(252 * train_years, len(feature_files) - 1)
    return feature_files[:split_idx], feature_files[split_idx:]