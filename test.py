from src import *

# Create features
with open('src/processed_data.pkl', 'rb') as f:
    op, high, low, close, volume, returns, risk_free_rate = pickle.load(f)
engineer = FeatureEngineer(op, high, low, close, volume, returns, risk_free_rate, lookback_days=65)
features = engineer.compute_all_features(save_path='src/data/')

