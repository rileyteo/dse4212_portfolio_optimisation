import pandas as pd
import pickle

with open('features_demo.pkl', 'rb') as f:
    features = pickle.load(f)
features = features.reset_index(drop=False)
print(features.index)