import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fredapi import Fred

import os
from dotenv import load_dotenv
from scipy.optimize import minimize
from scipy import stats

from sklearn.covariance import LedoitWolf

import pickle
import glob
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

