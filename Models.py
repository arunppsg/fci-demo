import streamlit as st 
import numpy as np 
import pandas as pd
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def remove_outliers(df, cols):
    for col in cols:
        df = df[np.abs(stats.zscore(df[col]) <= 3)].reset_index(drop=True)
    return df