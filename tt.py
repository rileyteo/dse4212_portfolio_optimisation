import pandas as pd
import pickle

with open('src/data_weekly_var/2013-06-03.pkl', 'rb') as f:
    features = pickle.load(f)
print(features.head())

