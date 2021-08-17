import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import Models
import graphs

st.title('''FCI COST FORECASTING
''')


vis = st.selectbox('Choose Visualization Type', ["Line Plot", "Table"])
Models.bplPopPlot(vis)

st.plotly_chart(graphs.get_food_subsidy_graph(), use_container_width=True)
st.plotly_chart(graphs.get_year_wise_total_ao_graph(), use_container_width=True)
