import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def get_food_subsidy_graph():
	df = pd.read_excel('src/data/Food Subsidy.xlsx')
	
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df['year'].astype(str), y=df['subsidy.released.for.current.year'], name='Subsidy Released',
	                         line=dict(width=4)))
	fig.add_trace(go.Scatter(x=df['year'].astype(str), y=df['subsidy.incurred.current.year'], name='Subsidy Incurred',
	                         line=dict(width=4)))
	fig.update_layout(
	    title={'text':'Subsidy Released For Current Year Vs Subsidy Incurred in Current Year'
	          },
	    xaxis_title="Financial Year",
	    yaxis_title="Rupees in crores",
	    legend_title="Legend",
	    autosize=True
	
	)
	fig.update_xaxes(type='category',
	                tickangle=45)

	return fig


def make_financial_year(year):
    fy = str(year) + '-' + str(str(year+1)[2:])
    return fy

def get_year_wise_total_ao_graph():
	df_rice = pd.read_excel('src/data/rice.xlsx')
	df_rice.drop(['State.UT', 'zone'], axis=1, inplace=True)
	df_rice = df_rice.groupby('year').agg(sum).reset_index().rename({'offtake':'rice_offtake', \
	                                                       'allotment':'rice_allotment'}, axis=1)
	
	df_wheat = pd.read_excel('src/data/wheat.xlsx')
	df_wheat.drop(['State.UT', 'zone'], axis=1, inplace=True)
	df_wheat = df_wheat.groupby('year').agg(sum).reset_index().rename({'offtake':'wheat_offtake',\
	                                                       'allotment':'wheat_allotment'}, axis=1)
	
	df_total = pd.merge(df_wheat, df_rice, on='year', how='inner')
	df_total['allotment'] = df_total['rice_allotment'] + df_total['wheat_allotment']
	df_total['offtake'] = df_total['rice_offtake'] + df_total['wheat_offtake']
	df_total.drop({'rice_allotment', 'wheat_allotment', 'rice_offtake', 'wheat_offtake'}, axis=1, inplace=True)
	
	
	df_total['year'] = df_total['year'].apply(make_financial_year)
	
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df_total['year'], y=df_total['allotment'], name='Allotment',
	                         line=dict(width=4)))
	fig.add_trace(go.Scatter(x=df_total['year'], y=df_total['offtake'], name='Offtake',
	                         line=dict(width=4)))
	fig.update_layout(
	    title={'text':'Total Allotment Vs Total Offtake over the years'
	          },
	    xaxis_title="Financial Year",
	    yaxis_title="Foodgrain Quanitiy in '000 MT '",
	    legend_title="Legend",
	    autosize=True
	
	)
	fig.update_xaxes(type='category',
	                tickangle=45)
	
	return fig	
