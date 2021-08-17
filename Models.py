

import streamlit as st 
import numpy as np 
import pandas as pd
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go


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

def all_pred_data(rp,bplChangeRate,pop,option,endYear,rice_bpl_fit,wheat_bpl_fit,rice_inc,wheat_inc):
    fut_data = load_pred_data(rp,bplChangeRate,pop,option,endYear)
    fut_data["Rice_Allotment"]=rice_bpl_fit.predict(fut_data[["Population","bpl_pop","rice_perc"]])
    fut_data["Wheat_Allotment"]=wheat_bpl_fit.predict(fut_data[["Population","bpl_pop","wheat_perc"]])

    if option=="ALL-INDIA":
        fut = fut_data.groupby(["year"]).sum()[['Rice_Allotment','Wheat_Allotment']].copy()
        fut["year"]=list(range(2020,endYear+1))
    else:
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
            fut["msp_rice"].iloc[i]= fut["msp_rice"].iloc[i-1]*(1+(rice_inc/100))
            fut["msp_wheat"].iloc[i]= fut["msp_wheat"].iloc[i-1]*(1+(wheat_inc/100))

    fut['cost']=(fut['msp_rice']*fut["Rice_Allotment"]+fut['msp_wheat']*fut['Wheat_Allotment'])*(10000/10000000)

    return fut 


def bplPopPlot(vis):
    

    st.sidebar.write('''
    ### Rice and Wheat Forecasts using Population and BPL model
    ''')
    

    bplChangeRate = st.sidebar.number_input('bpl change rate( in % )')
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

    vals=list(rw['State.UT'].unique())
    vals.append("ALL-INDIA")

    option = st.sidebar.selectbox('choose state',vals)
    
    rice_inc=st.sidebar.number_input('Rice MSP change rate (in %)')
    
    wheat_inc = st.sidebar.number_input('Wheat MSP change rate(in %)')

    endYear=st.sidebar.slider('Prediction upto (max year 2036)',2020,2036)
    
    st.write(f"""
        ### Rice and Wheat Forecasts for {option} from 2020 to {endYear}
        """)

    if vis == "Table":
        fut = all_pred_data(rp,bplChangeRate,pop,option,endYear,rice_bpl_fit,wheat_bpl_fit,rice_inc,wheat_inc)


        fut.rename({"year":"Year","msp_rice":"Rice_MSP","msp_wheat":"Wheat_MSP",
            "cost":"Procurement_Cost"}, 
            axis = "columns", inplace = True)

        st.dataframe(fut[["Year","Rice_Allotment","Wheat_Allotment","Rice_MSP","Wheat_MSP","Procurement_Cost"]])

    else:
        fig =  get_food_subsidy_graph_rice(all_pred_data(rp,bplChangeRate,pop,option,endYear,rice_bpl_fit,wheat_bpl_fit,rice_inc,wheat_inc),option,endYear)
        fig2 =  get_food_subsidy_graph_wheat(all_pred_data(rp,bplChangeRate,pop,option,endYear,rice_bpl_fit,wheat_bpl_fit,rice_inc,wheat_inc),option,endYear)

        st.plotly_chart(fig, use_container_width=True)

        st.plotly_chart(fig2, use_container_width=True)

    st.write(f''' 
        ### Models: 
        #### Rice_Allotment = C0population + C1bpl_population + C2rice_moving_perc + C3
        #### Wheat_Allotment = C0population + C1bpl_population + C2wheat_moving_perc + C3
        ### Units:
        #### Allotment unit is '000 Metric Tonnes
        #### MSP Price is for Per Qunital (INR)
        #### Procurement Cost is in Crores (INR)

        ''')

def get_food_subsidy_graph_rice(df,option,endYear):
	fig = go.Figure()

	fig.add_trace(go.Scatter(x=df['year'].astype(str), y=df['Rice_Allotment'], name='Rice Allotment',
	                         line=dict(width=4)))
	
	fig.update_layout(
	    title={'text':f'Rice Allotment Forecasts for {option} till {endYear}'
	          },
	    xaxis_title="Year",
	    yaxis_title="Allotment in '000 MTs",
	    legend_title="Legend",
	    autosize=True
	
	)
	fig.update_xaxes(type='category',
	                tickangle=45)

	return fig


def get_food_subsidy_graph_wheat(df,option,endYear):
	fig = go.Figure()

	fig.add_trace(go.Scatter(x=df['year'].astype(str), y=df['Wheat_Allotment'], name='Wheat Allotment',
	                         line=dict(width=4)))
	
	fig.update_layout(
	    title={'text':f'Wheat Allotment Forecasts for {option} till {endYear}'
	          },
	    xaxis_title="Year",
	    yaxis_title="Allotment in '000 MTs",
	    legend_title="Legend",
	    autosize=True
	
	)
	fig.update_xaxes(type='category',
	                tickangle=45)

	return fig
