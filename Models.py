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



def table():

    st.sidebar.write('''
    ### Rice and Wheat Forecasts using population model
    ''')
    pop = pd.read_excel("src/data/projected_population_by_state_2012_2036.xlsx")

    rw=load_rw()

    rp = pd.merge(rw, pop, on=['State.UT', 'year'], how='inner')
    rp = remove_outliers(rp, ["Population", "rice_allotment", "rice_moving_perc", "wheat_moving_perc"])


    rice_pop_fit = linear_model.LinearRegression().fit(rp[['Population', 'rice_moving_perc']], rp['rice_allotment'])


    wp = pd.merge(rw, pop, on=['State.UT', 'year'], how='inner')
    wp = remove_outliers(wp, ["Population", "wheat_allotment", "rice_moving_perc", "wheat_moving_perc"])
    wheat_pop_fit = linear_model.LinearRegression().fit(wp[['Population', 'wheat_moving_perc']], wp['wheat_allotment'])



    #prediction starts
   
    endYear2=st.sidebar.slider('Prediction upto (max year 2036)',2020,2036)

    historicAvg = rp.groupby("State.UT").mean()[["rice_perc","wheat_perc"]]

    pred_data=pd.merge(pop[pop["year"].between(2020,int(endYear2))], historicAvg, on=['State.UT'], how='inner')

    pred_data["rice_allotment"]=0
    pred_data["wheat_allotment"]=0


    pred_data["rice_allotment"] = rice_pop_fit.predict(pred_data[['Population','rice_perc']])
    pred_data["wheat_allotment"] = wheat_pop_fit.predict(pred_data[['Population','wheat_perc']])
    pred_data=pred_data.round(2)


    st.write("""
    ### Table 2
    """)
    st.write(f"""
    ## Rice Forecasts for 2020- {endYear2}
    """)

    st.write("""
    ### The Unit is '000 Metric Tonnes
    """)

    rice_allot=pred_data[["year","State.UT","rice_allotment"]]



    newf = rice_allot.pivot(index='State.UT', columns='year')
    newf.columns=list(range(2020,endYear2+1))
    newf[newf<0]=0

    st.dataframe(newf)



    st.write("""
    ### Table 3
    """)
    st.write(f"""
    ## Wheat Forecasts for 2020-{endYear2}
    """)

    st.write("""
    ### The Unit is '000 Metric Tonnes
    """)

    wheat_allot=pred_data[["year","State.UT","wheat_allotment"]]
    newf2 = wheat_allot.pivot(index='State.UT', columns='year')
    newf2.columns=list(range(2020,endYear2+1))
    newf2[newf2<0]=0

    st.dataframe(newf2)


    st.write(f"""
    ### Table 5.2 Rice and Wheat Procurement Forecasts for 2020 - {endYear2}
    """)

    st.write("""
    ### The Unit is '000 Metric Tonnes
    """)

    r=pred_data[["year","State.UT","rice_allotment","wheat_allotment"]].copy()
    r=r.groupby("year").sum().copy()
    r["total"]=r["rice_allotment"]+r["wheat_allotment"]

    r["msp_rice"]=0
    r["msp_wheat"]=0

    rice_inc=st.sidebar.number_input('Percentage change for MSP of rice')
    wheat_inc = st.sidebar.number_input('Percentage change for MSP of wheat')

    for i in range(0,(endYear2-2020)+1):
        if i==0:
            r["msp_rice"].iloc[0]=1868
            r["msp_wheat"].iloc[0]=1925
        elif i==1:
            r["msp_rice"].iloc[1]=1940
            r["msp_wheat"].iloc[1]=1975
        else:
            r["msp_rice"].iloc[i]= r["msp_rice"].iloc[i-1]*(1+(rice_inc/100))
            r["msp_wheat"].iloc[i]= r["msp_wheat"].iloc[i-1]*(1+(wheat_inc/100))

                                                    
    r['cost']=(r['msp_rice']*r["rice_allotment"]+r['msp_wheat']*r['wheat_allotment'])

    r.rename({"rice_allotment": "Rice Procurement", 
           "wheat_allotment": "Wheat Procurement"}, 
          axis = "columns", inplace = True)

    st.dataframe(r[["Rice Procurement","Wheat Procurement"]])



    st.write(f"""
    ### Table 5.3 Total Grain Procurement and Cost Forecasts for 2020 - {endYear2}
    """)

    r.rename({"total": "Total Grain Procurement (in '000 MTs)", 
           "cost": "Total Procurement Cost(in Crores)"}, 
          axis = "columns", inplace = True)
    st.dataframe(r[["Total Grain Procurement (in '000 MTs)","Total Procurement Cost(in Crores)"]])


def load_pred_data(rp,bplChangeRate,pop,option,endYear):
    
    future_bpl=rp[rp['year']==2019][["State.UT","bpl_pop","year"]]
    futurePopulation=pop[((pop['year']>=2019) & (pop['year']<=endYear))]
    fut_data = pd.merge(futurePopulation, future_bpl, on=['State.UT', 'year'], how='left')
    for year in range(2020, endYear+1):
        for state in list(fut_data['State.UT'].unique()):
            idx = fut_data[((fut_data['State.UT'] == state) & (fut_data['year'] == year))].index
            fut_data['bpl_pop'][idx]= (fut_data[((fut_data['State.UT'] == state) & (fut_data['year'] == year-1))]['bpl_pop'].values)*(1+bplChangeRate)  
    fut_data["rice_perc"]=rp[rp["State.UT"]==option]["rice_perc"].mean()
    fut_data["wheat_perc"]=rp[rp["State.UT"]==option]["wheat_perc"].mean()

    fut_data=fut_data[fut_data['year']>2019]
    fut_data=fut_data.fillna(0)
    return fut_data



def bplPlot():

    st.sidebar.write('''
    ### Rice and Wheat Forecasts using BPL model
    ''')
    
    bplChangeRate = st.sidebar.number_input('bpl change rate( in % )')
    
    bplChangeRate=bplChangeRate/100
    pop = pd.read_excel("src/data/projected_population_by_state_2012_2036.xlsx")
    bpl_perc2011 = pd.read_excel("src/data/BPL data.xlsx")
    bpl_perc2011.rename({"2011-12 Perc of Persons":"percent"}, axis=1, inplace=True)
    bpl_perc2011['year'] = 2011

    bpl = generate_bpl_data(pop, bpl_perc2011, bplChangeRate)


    rw=load_rw()


    rp = pd.merge(rw, bpl, on=['State.UT', 'year'], how='inner')

    rp = remove_outliers(rp, ["bpl_pop", "rice_allotment", "rice_moving_perc", "wheat_moving_perc"])


    rice_bpl_fit = linear_model.LinearRegression().fit(rp[['bpl_pop', 'rice_moving_perc']], rp['rice_allotment'])

    wheat_bpl_fit = linear_model.LinearRegression().fit(rp[['bpl_pop', 'wheat_moving_perc']], rp['wheat_allotment'])

    #prediction
    

    option = st.sidebar.selectbox('choose state',list(rw['State.UT'].unique()))

    endYear=st.sidebar.slider('Prediction upto (max year 2036)',2020,2036)
   
    fut_data = load_pred_data(rp,bplChangeRate,pop,option,endYear)

    fut_data["Rice_Allotment"]=rice_bpl_fit.predict(fut_data[["bpl_pop","rice_perc"]])
    fut_data["Wheat_Allotment"]=wheat_bpl_fit.predict(fut_data[["bpl_pop","wheat_perc"]])
    fut=fut_data[fut_data['State.UT']==option][['year','Rice_Allotment','Wheat_Allotment']].copy()
    fut[fut<0]=0
    fut=fut.round(2)

    

    st.write(f"""
    ### Rice and Wheat Forecasts for {option} from 2020 to {endYear}
    """)
  
    st.write("""
    ### The Unit is '000 Metric Tonnes
    """)

    st.dataframe(fut[["year","Rice_Allotment","Wheat_Allotment"]])




def bplPopPlot():
    
    st.sidebar.write('''
    ### Rice and Wheat Forecasts using Population and BPL model
    ''')
    
    bplChangeRate = st.sidebar.number_input('bpl change rate( in % )')
    rice_inc2=st.sidebar.number_input('Percentage change for MSP of rice')
    wheat_inc2 = st.sidebar.number_input('Percentage change for MSP of wheat')
    
    pop = pd.read_excel("src/data/projected_population_by_state_2012_2036.xlsx")
    bpl_perc2011 = pd.read_excel("src/data/BPL data.xlsx")
    bpl_perc2011.rename({"2011-12 Perc of Persons":"percent"}, axis=1, inplace=True)
    bpl_perc2011['year'] = 2011

    bpl = generate_bpl_data(pop, bpl_perc2011, bplChangeRate)
    
    rw=load_rw()

    rp = pd.merge(rw, bpl, on=['State.UT', 'year'], how='inner')
    
    rp = remove_outliers(rp, ["Population","bpl_pop", "rice_allotment", "rice_moving_perc", "wheat_moving_perc"])

    rice_bpl_fit = linear_model.LinearRegression().fit(rp[['Population','bpl_pop', 'rice_moving_perc']], rp['rice_allotment'])

    wheat_bpl_fit = linear_model.LinearRegression().fit(rp[['Population','bpl_pop', 'wheat_moving_perc']], rp['wheat_allotment'])

    #prediction
  


    option = st.sidebar.selectbox('choose state',list(rw['State.UT'].unique()))

    endYear=st.sidebar.slider('Prediction upto (max year 2036)',2020,2036)
    
    

    fut_data = load_pred_data(rp,bplChangeRate,pop,option,endYear)


    fut_data["Rice_Allotment"]=rice_bpl_fit.predict(fut_data[["Population","bpl_pop","rice_perc"]])
    fut_data["Wheat_Allotment"]=wheat_bpl_fit.predict(fut_data[["Population","bpl_pop","wheat_perc"]])
    fut=fut_data[fut_data['State.UT']==option][['year','Rice_Allotment','Wheat_Allotment']].copy()
    fut[fut<0]=0
    fut=fut.round(2)

    
    fut["msp_rice"]=0
    fut["msp_wheat"]=0

    for i in range(0,(endYear-2020)+1):
        if i==0:
            fut["msp_rice"].iloc[0]=1868
            fut["msp_wheat"].iloc[0]=1925
        elif i==1:
            fut["msp_rice"].iloc[1]=1940
            fut["msp_wheat"].iloc[1]=1975
        else:
            fut["msp_rice"].iloc[i]= fut["msp_rice"].iloc[i-1]*(1+(rice_inc2/100))
            fut["msp_wheat"].iloc[i]= fut["msp_wheat"].iloc[i-1]*(1+(wheat_inc2/100))

                                                    
    fut['cost']=(fut['msp_rice']*fut["rice_allotment"]*10000) + (fut['msp_wheat']*fut['wheat_allotment']*10000)

     
    st.write(f"""
    ### Rice and Wheat Forecasts for {option} from 2020 to {endYear}
    """)
  
    st.write("""
    ### The Unit is '000 Metric Tonnes
    """)

    st.dataframe(fut[["year","Rice_Allotment","Wheat_Allotment","msp_rice","msp_wheat","cost"]])

