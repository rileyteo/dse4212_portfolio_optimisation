import pandas as pd
import pickle

with open('src/data_monthly_ret/2013-06-03.pkl', 'rb') as f:
    features = pickle.load(f)
    print(features.iloc[1])
    print(features.shape)
with open('src/data_monthly_var/2013-06-03.pkl', 'rb') as f:
    features = pickle.load(f)
    print(features.iloc[1])
    print(features.shape)
with open('src/data_weekly_var/2013-06-03.pkl', 'rb') as f:
    features = pickle.load(f)
    print(features.iloc[1])
    print(features.shape)
with open('src/data_weekly_ret/2013-06-03.pkl', 'rb') as f:
    features = pickle.load(f)
    print(features.iloc[1])
    print(features.shape)


