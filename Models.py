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



def generate_bpl_data(pop, bpl, bpl_cr):
    states = bpl['State.UT'].unique()
    states
    for state in states:
        perc = bpl[(bpl['State.UT'] == state) & (bpl['year'] == 2011)]['percent'].values[0]
        for year in range(2012, 2020):
            perc = perc + bpl_cr
            new_entry = pd.DataFrame({'State.UT':[state], 'percent':[perc], 'year':[year]})
            bpl = pd.concat([bpl, new_entry], axis=0)

    state = "ANDHRA PR"

    bpl = bpl[~((bpl['year'] > 2013) & (bpl['State.UT'] == state))]

    bpl = pd.merge(bpl, pop, on=['State.UT', 'year'])
    bpl['bpl_pop'] = bpl['percent'] * bpl['Population'] / 100
    bpl = bpl[(bpl['bpl_pop'] > 0)]
    bpl['log_bpl_pop'] = np.log1p(bpl['Population'])
    return bpl


def load_rw():
    rice = pd.read_excel("src/data/rice.xlsx")
    wheat = pd.read_excel("src/data/wheat.xlsx")
    #print (rice.shape)

    rice = remove_outliers(rice, ["offtake", "allotment"])
    wheat = remove_outliers(wheat, ["offtake", "allotment"])
    #print (rice.shape)

    r = rice.copy()
    w = wheat.copy()
    r.rename({"allotment":"rice_allotment"}, axis=1, inplace=True)
    w.rename({"allotment":"wheat_allotment"}, axis=1, inplace=True)
    r.drop(["zone", "offtake"], axis=1, inplace=True)
    w.drop(["zone", "offtake"], axis=1, inplace=True)
    rw = pd.merge(r, w, on=['State.UT', 'year'], how='inner')

    rw['rice_perc'] = rw['rice_allotment'] / (rw['rice_allotment'] + rw['wheat_allotment'])
    rw['wheat_perc'] = rw['wheat_allotment'] / (rw['rice_allotment'] + rw['wheat_allotment'])


    rw['rice_moving_perc'] = 0
    rw['wheat_moving_perc'] = 0


    for year in range(2006, 2020):
        for state in list(rw['State.UT'].unique()):
            df2 = rw[((rw['State.UT'] == state) & ((rw['year'] < year) & (rw['year'] >= year-3)))]
            r_m_p, w_m_p = df2['rice_perc'].mean(), df2['wheat_perc'].mean()
            idx = rw[((rw['State.UT'] == state) & (rw['year'] == year))].index
            if len(idx) > 0:
                rw['rice_moving_perc'][idx] = r_m_p
                rw['wheat_moving_perc'][idx] = w_m_p

    rw = rw[(rw['rice_moving_perc'] > 0) & (rw['wheat_moving_perc'] > 0)]
    return rw

