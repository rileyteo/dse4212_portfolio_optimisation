import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fredapi import Fred

import os
from dotenv import load_dotenv
from scipy.optimize import minimize

from sklearn.covariance import LedoitWolf
