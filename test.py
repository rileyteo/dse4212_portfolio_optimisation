from src import *
from multiprocessing import Process

# Load data
with open('src/processed_data.pkl', 'rb') as f:
    op, high, low, close, volume, returns, risk_free_rate = pickle.load(f)

# Create engineer
engineer = FeatureEngineer(op, high, low, close, volume, returns, risk_free_rate, lookback_days=70)

# Define functions
def func1():
    features, df = engineer.compute_monthly_volatility_features('src/data_monthly_var/')
    print(f"Monthly vol: {features.shape}, targets: {df.shape}")

def func3():
    features, df = engineer.compute_monthly_return_features('src/data_monthly_ret/')
    print(f"Monthly ret: {features.shape}, targets: {df.shape}")

def func4():
    features, df = engineer.compute_weekly_return_features('src/data_weekly_ret/')
    print(f"Weekly ret: {features.shape}, targets: {df.shape}")

def func5():
    features, df = engineer.compute_weekly_volatility_features('src/data_weekly_var/')
    print(f"Weekly vol: {features.shape}, targets: {df.shape}")

if __name__ == '__main__':  # IMPORTANT: Protect entry point
    # Create processes
    p1 = Process(target=func1)
    p3 = Process(target=func3)
    p4 = Process(target=func4)
    p5 = Process(target=func5)
    
    # Start all
    p1.start()
    p3.start()
    p4.start()
    p5.start()
    
    # Wait for completion
    p1.join()
    p3.join()
    p4.join()
    p5.join()
    
    print("✓ All feature engineering complete!")