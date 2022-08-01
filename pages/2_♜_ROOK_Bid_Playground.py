import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import date

from models.rook_bid import RookBidModel
from util.params import *

st.set_page_config(layout="wide")

sim_length = 365 * 10

# Set General Protocol and Ecosystem params
with st.expander("Protocol and Ecosystem Parameters", expanded=True):
    target_bid_percentage = st.slider(
        "Target Keeper Bid Ratio",
        help="The ratio of the $ value of winning bid to the $ value of the profit opportunity on average. \
            Adjustable by tuning the greenlight algorithm",
        min_value=0.5,
        max_value=0.99,
        value=0.8,
        step=0.01,
    )
    mev_volume_ratio = st.slider(
        "MEV / Volume Ratio",
        help="The ratio of MEV captured by Keepers to overall trading volume through the coordinator. \
            Since launch we've observed a consistent ratio of 0.001, but this could change with swaps / integrations",
        min_value=0.0005,
        max_value=0.0015,
        value=0.001,
        step=0.0001,
        format="%.4f",
    )
    user_claim_percentage = st.slider(
        "User Claim Ratio",
        help="The percentage of rewards claimed by users, on average. \
            In this model, all user claimed rewards are assumed to be sold immediately",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
    )
    partner_claim_percentage = st.slider(
        "Partner Claim Ratio",
        help="The percentage of rewards claimed by partners, on average. \
            In this model, all partner claimed rewards are assumed to be sold immediately",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
    )
    treasury_burn_rate = st.number_input(
        "Daily Treasury Expenses in $",
        help="The daily treasury burn rate in USD. This includes contributor salaries, \
            software / infra costs, etc",
        min_value=20000,
        max_value=50000,
        value=25000,
    )
    initial_rook_price = st.number_input(
        "Initial ROOK Price", help="The price of ROOK in $ to start the simulation with", value=40
    )

# Set bid distribution params
with st.expander("Bid Distribution Parameters", expanded=True):
    bid_input_container, bid_pie_container = st.columns(spec=2)
    treasury_bid = bid_input_container.slider(label="Treasury", min_value=0.0, max_value=0.15, value=0.06, step=0.01)
    partner_bid = bid_input_container.slider(label="Partners", min_value=0.0, max_value=0.15, value=0.0, step=0.01)
    burn_bid = bid_input_container.slider(label="Burn", min_value=0.0, max_value=0.1, value=0.04, step=0.01)
    stake_bid = bid_input_container.slider(label="Staking", min_value=0.0, max_value=0.1, value=0.1, step=0.01)
    user_bid = 1 - treasury_bid - partner_bid - burn_bid - stake_bid

    fig = go.Figure(
        go.Pie(
            labels=["Treasury", "Partners", "Burn", "Stakers", "Users"],
            values=[treasury_bid, partner_bid, burn_bid, stake_bid, user_bid],
            hoverinfo="label+percent",
            textinfo="label",
        )
    )
    bid_pie_container.plotly_chart(fig, use_container_width=True)

# Set volume model
with st.expander("Volume Growth Model", expanded=True):
    today = date.today()
    volume_model = st.selectbox(
        label="Growth Model",
        help="The function used to model volume growth over the course of the simulation.",
        options=("logistic", "constant", "linear"),
    )
    volume_param_selector, volume_chart = st.columns(spec=2)

    start_volume = 0
    max_volume = 0
    volume_growth_rate = 0

    if volume_model == "constant":
        start_volume = volume_param_selector.number_input(
            "Daily Volume in $", min_value=100000, max_value=500000000, value=1000000
        )

        volume_df = pd.DataFrame(
            {
                "Date": pd.Series(pd.date_range(today, periods=sim_length, freq="D")),
                "Volume": np.full(sim_length, start_volume),
            }
        )

    elif volume_model == "linear":
        start_volume = volume_param_selector.number_input(
            "Starting Volume in $", min_value=100000, max_value=500000000, value=500000
        )
        max_volume = volume_param_selector.number_input(
            "End Volume in $", min_value=100000, max_value=500000000, value=100000000
        )

        m = (max_volume - start_volume) / sim_length
        x = np.arange(sim_length)
        y = m * x + start_volume

        volume_df = pd.DataFrame({"Date": pd.Series(pd.date_range(today, periods=sim_length, freq="D")), "Volume": y})

    elif volume_model == "logistic":
        start_volume = volume_param_selector.number_input(
            "Starting Volume in $", min_value=100000, max_value=5000000, value=1000000
        )
        max_volume = volume_param_selector.number_input(
            "Max Volume in $", min_value=10000000, max_value=500000000, value=200000000
        )
        volume_growth_rate = volume_param_selector.slider(
            "Volume Growth Rate", min_value=0.002, max_value=0.005, value=0.0035, step=0.0001, format="%.4f"
        )

        p0 = start_volume
        k = max_volume
        r = volume_growth_rate
        exp = np.exp(r * np.arange(0, sim_length))
        volume_timeseries = (k * exp * p0) / (k + (exp - 1) * p0)

        print(volume_timeseries)

        volume_df = pd.DataFrame(
            {"Date": pd.Series(pd.date_range(today, periods=sim_length, freq="D")), "Volume": volume_timeseries}
        )

    fig = go.Figure(
        go.Scatter(
            x=volume_df["Date"],
            y=volume_df["Volume"],
        )
    )

    fig.update_layout(
        title="Daily Volume in $",
        xaxis_title="Date",
        yaxis_title="Volume",
    )

    volume_chart.plotly_chart(fig, use_container_width=True)

# Set liquidity model
with st.expander("Liquidity Model", expanded=True):
    liquidity_model = st.selectbox(
        label="Liquidity Model",
        help="How to model ROOK liquidity. This simulation assumes all ROOK liquidity is in a ROOK/USDC pool \
            in constant-product AMM like uniswap v2 or sushiswap. Constant means a fixed $ amount of ROOK \
            liquidity over the course of the simulation. mcap means the liquidity is modeled as a fraction \
            of ROOK market cap (circulating supply * price). supply means liquidity is modeled as a \
            fraction of the circulating supply of ROOK, with a matching USDC amount. \
            Liquidity in ROOK anbd USDC units is recalculated at the beginning of each timestep ",
        options=("mcap", "constant", "supply"),
    )

    if liquidity_model == "constant":
        liquidity_constant = st.number_input("AMM Liquidity in $", min_value=1000000, max_value=50000000, value=5000000)
    elif liquidity_model == "mcap":
        liquidity_constant = st.slider("AMM Liquidity as % of $ market cap", min_value=0.01, max_value=0.2, value=0.1)
    elif liquidity_model == "supply":
        liquidity_constant = st.slider(
            "AMM Liquidity as % of ROOK circ supply", min_value=0.01, max_value=0.2, value=0.1
        )

# Calculate model
bid_params = BidDistributionParams(treasury_bid, partner_bid, stake_bid, burn_bid)
protocol_params = ProtocolParams(target_bid_percentage, bid_params)
ecosystem_params = EcosystemParams(mev_volume_ratio, user_claim_percentage, partner_claim_percentage)
dao_params = DAOParams(treasury_burn_rate)
volume_params = VolumeParams(start_volume, volume_growth_rate, max_volume)

model = RookBidModel(
    sim_length_days=sim_length,
    protocol_params=protocol_params,
    bid_distribution_params=bid_params,
    ecosystem_params=ecosystem_params,
    dao_params=dao_params,
    volume_model=volume_model,
    volume_params=volume_params,
    liquidity_model=liquidity_model,
    liquidity_constant=liquidity_constant,
    initial_rook_price=initial_rook_price,
    treasury_stables=27000000,
)

df = model.run_sim()

circ_supply_timeseries = (
    model.rook_supply.total_supply
    - model.rook_supply.strategic_reserves
    - df["treasury_rook"]
    - df["unclaimed_rook"]
    - df["burned_rook"]
)

mcap_timeseries = circ_supply_timeseries * df["rook_price"]

fig = make_subplots(
    rows=4,
    cols=2,
    subplot_titles=(
        "ROOK Price",
        "Staked ROOK",
        "Treasury ROOK balance",
        "Unclaimed ROOK",
        "Daily CG Volume",
        "Burned ROOK",
        "Circulating Supply",
        "Market Cap",
    ),
)

fig.append_trace(go.Scatter(x=df["date"], y=df["rook_price"], name="ROOK Price"), row=1, col=1)

fig.append_trace(go.Scatter(x=df["date"], y=df["treasury_rook"], name="Treasury ROOK balance"), row=2, col=1)

fig.append_trace(go.Scatter(x=df["date"], y=df["daily_volume"], name="Daily Volume"), row=3, col=1)

fig.append_trace(go.Scatter(x=df["date"], y=circ_supply_timeseries, name="Circ Supply"), row=4, col=1)

fig.append_trace(go.Scatter(x=df["date"], y=df["staked_rook"], name="Staked ROOK"), row=1, col=2)

fig.append_trace(go.Scatter(x=df["date"], y=df["unclaimed_rook"], name="Unclaimed ROOK"), row=2, col=2)

fig.append_trace(go.Scatter(x=df["date"], y=df["burned_rook"], name="Burned ROOK"), row=3, col=2)

fig.append_trace(go.Scatter(x=df["date"], y=mcap_timeseries, name="Market Cap"), row=4, col=2)

fig.update_layout(height=1000)

st.plotly_chart(fig, use_container_width=True)
