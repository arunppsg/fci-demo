import streamlit as st
import numpy as np
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import joblib

import pandas as pd

# For hiding SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'


def remove_outliers(df, cols):
    # Outliers are points which are more than 3 standard deviations from
    # the mean
    for col in cols:
        df = df[np.abs(stats.zscore(df[col]) <= 3)].reset_index(drop=True)
    return df


def load_rw():
    rice = pd.read_excel("src/data/rice.xlsx")
    wheat = pd.read_excel("src/data/wheat.xlsx")
    # print (rice.shape)

    #    rice = remove_outliers(rice, ["offtake", "allotment"])
    #    wheat = remove_outliers(wheat, ["offtake", "allotment"])
    # print (rice.shape)

    r = rice.copy()
    w = wheat.copy()
    r.rename({"allotment": "rice_allotment"}, axis=1, inplace=True)
    w.rename({"allotment": "wheat_allotment"}, axis=1, inplace=True)
    r.drop(["zone", "offtake"], axis=1, inplace=True)
    w.drop(["zone", "offtake"], axis=1, inplace=True)
    rw = pd.merge(r, w, on=["State.UT", "year"], how="inner")

    rw["rice_perc"] = rw["rice_allotment"] / (
        rw["rice_allotment"] + rw["wheat_allotment"]
    )
    rw["wheat_perc"] = rw["wheat_allotment"] / (
        rw["rice_allotment"] + rw["wheat_allotment"]
    )

    rw["rice_moving_perc"] = 0
    rw["wheat_moving_perc"] = 0

    for year in range(2006, 2020):
        for state in list(rw["State.UT"].unique()):
            df2 = rw[
                (
                    (rw["State.UT"] == state)
                    & ((rw["year"] < year) & (rw["year"] >= year - 3))
                )
            ]
            r_m_p, w_m_p = df2["rice_perc"].mean(), df2["wheat_perc"].mean()
            idx = rw[((rw["State.UT"] == state) & (rw["year"] == year))].index
            if len(idx) > 0:
                rw["rice_moving_perc"][idx] = r_m_p
                rw["wheat_moving_perc"][idx] = w_m_p

    rw = rw[(rw["rice_moving_perc"] > 0) & (rw["wheat_moving_perc"] > 0)]
    return rw


def generate_pred_data(rp, bpl_change_rate, pop, option, endYear):
    # Generates prediction datapoints
    future_bpl = rp[rp["year"] == 2019][["State.UT", "bpl_pop", "year"]]
    future_population = pop[((pop["year"] >= 2019) & (pop["year"] <= endYear))]
    fut_data = pd.merge(
        future_population, future_bpl, on=["State.UT", "year"], how="left"
    )
    for year in range(2020, endYear + 1):
        for state in list(fut_data["State.UT"].unique()):
            idx = fut_data[
                ((fut_data["State.UT"] == state) & (fut_data["year"] == year))
            ].index
            fut_data["bpl_pop"][idx] = (
                fut_data[
                    ((fut_data["State.UT"] == state) & (fut_data["year"] == year - 1))
                ]["bpl_pop"].values
            ) * (
                1 + bpl_change_rate / 100
            )  # bpl change rate is percentage, hence divide by 100

    rice_wheat_perc_mean = (
        rp.groupby("State.UT")
        .agg({"rice_perc": np.mean, "wheat_perc": np.mean})
        .reset_index()
    )
    fut_data = pd.merge(fut_data, rice_wheat_perc_mean, on=["State.UT"], how="left")
    fut_data = fut_data.rename(
        {"rice_perc": "rice_moving_perc", "wheat_perc": "wheat_moving_perc"}, axis=1
    )

    # 2020 because the year we have values till 2019-20.
    # The year in data represents the financial year start.
    fut_data = fut_data[fut_data["year"] > 2020]
    fut_data = fut_data.fillna(0)
    return fut_data


def all_pred_data(
    rp,
    bpl_change_rate,
    pop,
    option,
    endYear,
    rice_inc,
    wheat_inc,
):
    fut_data = generate_pred_data(rp, bpl_change_rate, pop, option, endYear)
    rice_bpl_fit = joblib.load("src/models/rice_bpl_pop_fit.joblib")
    wheat_bpl_fit = joblib.load("src/models/wheat_bpl_pop_fit.joblib")
    fut_data["Rice_Allotment"] = rice_bpl_fit.predict(
        fut_data[["Population", "bpl_pop", "rice_moving_perc"]]
    )
    fut_data["Wheat_Allotment"] = wheat_bpl_fit.predict(
        fut_data[["Population", "bpl_pop", "wheat_moving_perc"]]
    )

    if option == "ALL-INDIA":
        fut = (
            fut_data.groupby(["year"])
            .sum()[["Rice_Allotment", "Wheat_Allotment"]]
            .reset_index()
            .copy()
        )
        fut["year"] = list(range(2021, endYear + 1))
    else:
        fut = fut_data[fut_data["State.UT"] == option][
            ["year", "Rice_Allotment", "Wheat_Allotment"]
        ].copy()
    fut[fut < 0] = 0
    fut = fut.round(2)

    fut["msp_rice"] = 0
    fut["msp_wheat"] = 0

    for i in range(0, (endYear - 2021) + 1):
        if i == 0:
            # The actual minimum support price for the year 2020-21
            fut["msp_rice"].iloc[0] = 1868
            fut["msp_wheat"].iloc[0] = 1925
        elif i == 1:
            # The actual msp for the year 2021-22
            fut["msp_rice"].iloc[1] = 1940
            fut["msp_wheat"].iloc[1] = 1975
        else:
            fut["msp_rice"].iloc[i] = fut["msp_rice"].iloc[i - 1] * (
                1 + (rice_inc / 100)
            )
            fut["msp_wheat"].iloc[i] = fut["msp_wheat"].iloc[i - 1] * (
                1 + (wheat_inc / 100)
            )

    fut["Rice_Allotment"] = (
        fut["Rice_Allotment"] * 1000 * 1000
    )  # converting into kg from '000 MT
    fut["Wheat_Allotment"] = fut["Wheat_Allotment"] * 1000 * 1000
    fut["msp_rice"] = fut["msp_rice"] / 100  # Converting to per kg
    fut["msp_wheat"] = fut["msp_wheat"] / 100  # Converting to per kg
    fut["cost"] = (
        fut["msp_rice"] * fut["Rice_Allotment"]
        + fut["msp_wheat"] * fut["Wheat_Allotment"]
    )

    return fut


def bpl_population_plot(vis):
    # Entry point
    st.sidebar.write(
        """
    ### Rice and Wheat Forecasts
    """
    )

    bpl_change_rate = st.sidebar.number_input("BPL Change Rate(in %)")
    pop = pd.read_excel("src/data/projected_population_by_state_2012_2036.xlsx")
    bpl = pd.read_csv("src/data/bpl_2011_19_cr_least_mse_fit.csv")

    rw = load_rw()
    # rice wheat population data
    rp = pd.merge(rw, bpl, on=["State.UT", "year"], how="inner")

    vals = list(rw["State.UT"].unique())
    vals.insert(0, "ALL-INDIA")

    option = st.sidebar.selectbox("State", vals)
    rice_msp_inc = st.sidebar.number_input("Rice MSP Change Rate (in %)")
    wheat_msp_inc = st.sidebar.number_input("Wheat MSP Change Rate(in %)")
    endYear = st.sidebar.slider("Prediction upto (max year 2036)", 2021, 2036)

    st.write(
        f"""
        ### Rice and Wheat Forecasts for {option} from 2021 to {endYear}
        """
    )

    fut = all_pred_data(
        rp,
        bpl_change_rate,
        pop,
        option,
        endYear,
        rice_msp_inc,
        wheat_msp_inc,
    )
    fut.rename(
        {
            "year": "Year",
            "msp_rice": "Rice_MSP",
            "msp_wheat": "Wheat_MSP",
            "cost": "Total_Procurement_Cost",
        },
        axis="columns",
        inplace=True,
    )

    if vis == "Table":
        # st.dataframe(fut[["Year","Rice_Allotment","Wheat_Allotment","Rice_MSP","Wheat_MSP","Total_Procurement_Cost"]])
        st.dataframe(
            fut[["Year", "Rice_Allotment", "Wheat_Allotment", "Total_Procurement_Cost"]]
        )

    else:
        fig = get_food_subsidy_graph_rice(fut, option, endYear)
        fig2 = get_food_subsidy_graph_wheat(fut, option, endYear)

        st.plotly_chart(fig, use_container_width=True)

        st.plotly_chart(fig2, use_container_width=True)

        total_cost_fig = get_total_procurement_cost(
            fut[["Year", "Total_Procurement_Cost"]], option, endYear
        )
        st.plotly_chart(total_cost_fig, use_container_width=True)

    st.write(
        f"""
        ### Prediction Units:
        Allotment - '000 Metric Tonnes;
		
        Procurement Cost  - Crores (INR)
        """
    )

    st.write(
        f"""
		### Model used for rice prediction
		$rice\_allotment = C_0 population + C_1 bpl\_population + C_2 rice\_moving\_perc + C_3$
		"""
    )

    st.write(
        f"""
		### Model used for wheat prediction
		$wheat\_allotment = D_0 population + D_1 bpl\_population + D_2 wheat\_moving\_perc + D_3$
		"""
    )


def get_food_subsidy_graph_rice(df, option, endYear):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Year"].astype(str),
            y=df["Rice_Allotment"],
            name="Rice Allotment",
            line=dict(width=4),
        )
    )

    fig.update_layout(
        title={
            "text": f"Rice Allotment Forecasts for {option} from 2021 till {endYear}"
        },
        xaxis_title="Year",
        yaxis_title="Allotment in '000 MTs",
        legend_title="Legend",
        autosize=True,
    )
    fig.update_xaxes(type="category", tickangle=45)

    return fig


def get_food_subsidy_graph_wheat(df, option, endYear):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Year"].astype(str),
            y=df["Wheat_Allotment"],
            name="Wheat Allotment",
            line=dict(width=4),
        )
    )

    fig.update_layout(
        title={
            "text": f"Wheat Allotment Forecasts for {option} from 2021 till {endYear}"
        },
        xaxis_title="Year",
        yaxis_title="Allotment in '000 MTs",
        legend_title="Legend",
        autosize=True,
    )
    fig.update_xaxes(type="category", tickangle=45)

    return fig


def get_total_procurement_cost(df, option, endYear):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Year"].astype(str),
            y=df["Total_Procurement_Cost"],
            name="Procurement Cost",
            line=dict(width=4),
        )
    )
    fig.update_layout(
        title={
            "text": f"Total Procurement Costs of Rice and Wheat for {option} from 2021 till {endYear}"
        },
        xaxis_title="Year",
        yaxis_title="Cost in Rs. Crores",
        legend_title="Legend",
        autosize=True,
    )
    fig.update_xaxes(type="category", tickangle=45)
    return fig
