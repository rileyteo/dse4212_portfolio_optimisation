from src import *

# Create features
with open('src/processed_data.pkl', 'rb') as f:
    op, high, low, close, volume, returns, risk_free_rate = pickle.load(f)
engineer = FeatureEngineer(op, high, low, close, volume, returns, risk_free_rate, lookback_days=70)
# features = engineer.compute_all_features(save_path='src/data/')
features, df = engineer.compute_all_features_volatility_weekly('src/data_weekly_var/')
df.to_pickle('src/data_weekly_var/weekly_volatility_targets.pkl')

