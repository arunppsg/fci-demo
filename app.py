import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import graphs
import new
import Models

st.title('''FCI COST FORECASTING
''')

#st.plotly_chart(graphs.get_food_subsidy_graph(), use_container_width=True)
#st.plotly_chart(graphs.get_year_wise_total_ao_graph(), use_container_width=True)

context = st.selectbox('choose model', ["pop+moving_perc","bpl+moving_perc","pop+bpl_moving_perc"])

if context == "pop+moving_perc":
    Models.table()
elif context =="bpl+moving_perc":
    Models.bplPlot()
else:
    Models.bplPopPlot()