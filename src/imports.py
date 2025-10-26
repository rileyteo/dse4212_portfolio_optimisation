import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from fredapi import Fred

import os
from dotenv import load_dotenv
from scipy.optimize import minimize
from scipy import stats
from scipy.stats import spearmanr

from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import pickle
import glob
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


from sklearn.metrics import r2_score

from joblib import Parallel, delayed
from tqdm.auto import tqdm

from itertools import product