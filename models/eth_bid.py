import math
import numpy as np
import pandas as pd

from datetime import date, datetime

from util.balances import RookSupply
from util.params import *


class EthBidModel:
    def __init__(
        self,
        sim_length_days: int,
        protocol_params: ProtocolParams,
        bid_distribution_params: BidDistributionParams,
        ecosystem_params: EcosystemParams,
        dao_params: DAOParams,
        volume_model: str,
        volume_params: VolumeParams,
        liquidity_model: str,
        liquidity_constant: float,
        initial_rook_price: float,
        treasury_stables: float = 27000000,
    ):

        # Set model parameters
        self.sim_length_days = sim_length_days
        self.bid_distribution_params = bid_distribution_params
        self.protocol_params = protocol_params
        self.ecosystem_params = ecosystem_params
        self.dao_params = dao_params
        self.volume_model = volume_model
        self.volume_params = volume_params
        self.liquidity_model = liquidity_model
        self.liquidity_constant = liquidity_constant

        # Set initial conditions
        self.rook_supply = RookSupply()
        self.rook_price = initial_rook_price
        self.treasury_stables = treasury_stables
        self.treasury_eth = 10000
        self.eth_price = 2000

        # init timeseries for model outputs
        self.rook_price_timeseries = [self.rook_price]
        self.staked_rook_timeseries = [self.rook_supply.staked]
        self.treasury_rook_timeseries = [self.rook_supply.treasury]
        self.unclaimed_rook_timeseries = [self.rook_supply.unclaimed]
        self.burned_rook_timeseries = [self.rook_supply.burned]
        self.treasury_eth_timeseries = [self.treasury_eth]

        if self.volume_model == "constant":
            self.volume_timeseries = np.full(self.sim_length_days, self.volume_params.start_volume)
        elif self.volume_model == "linear":
            x = np.arange(self.sim_length_days)
            m = (self.volume_params.max_volume - self.volume_params.start_volume) / self.sim_length_days
            y = m * x + self.volume_params.start_volume
            self.volume_timeseries = y
        else:  # logistic
            p0 = self.volume_params.start_volume
            k = self.volume_params.max_volume
            r = self.volume_params.volume_growth_rate
            exp = np.exp(r * np.arange(0, sim_length_days))
            self.volume_timeseries = (k * exp * p0) / (k + (exp - 1) * p0)

    def iterate_one_day(self, volume_usd: float, treasury_burn: bool):

        # Set AMM Liquidity:
        if self.liquidity_model == "mcap":
            market_cap = self.rook_supply.get_circulating_supply() * self.rook_price
            amm_usdc = self.liquidity_constant * market_cap
            amm_rook = amm_usdc / self.rook_price
        elif self.liquidity_model == "supply":
            amm_rook = self.liquidity_constant * self.rook_supply.get_circulating_supply()
            amm_usdc = amm_rook * self.rook_price
        else:
            amm_usdc = self.liquidity_constant / 2
            amm_rook = amm_usdc / self.rook_price

        # STEP 1: Keepers buying ETH to bid:
        daily_bid_volume_usd = (
            volume_usd * self.ecosystem_params.mev_volume_ratio * self.protocol_params.target_bid_percent
        )
        keeper_eth_bought = daily_bid_volume_usd / self.eth_price

        # STEP 2: Keepers bid ETH:
        user_bid = keeper_eth_bought * self.bid_distribution_params.user
        treasury_bid = keeper_eth_bought * self.bid_distribution_params.treasury
        partner_bid = keeper_eth_bought * self.bid_distribution_params.partner
        burn_bid = keeper_eth_bought * self.bid_distribution_params.burn
        stake_bid = keeper_eth_bought * self.bid_distribution_params.stake

        # STEP 3: Burn cut gets bought off market (assume lossless eth-usdc conversion):
        burn_rook_bought = burn_bid * (self.eth_price / self.rook_price)
        burn_usdc_sold = (amm_usdc * burn_rook_bought) / (amm_rook - burn_rook_bought)

        # New AMM pool balances
        amm_rook -= burn_rook_bought
        amm_usdc += burn_usdc_sold

        # STEP 5: Treasury sells ETH
        if treasury_burn:
            treasury_eth_sold = self.dao_params.daily_treasury_burn / self.eth_price
        else:
            treasury_eth_sold = 0

        # Update ROOK price and supply balances
        self.treasury_eth += treasury_bid - treasury_eth_sold
        self.rook_price = amm_usdc / amm_rook
        self.rook_supply.burned += burn_bid

    def run_sim(self):

        # days before the treasury must start selling ROOK
        treasury_burn = False
        stable_runway = math.floor(self.treasury_stables / self.dao_params.daily_treasury_burn)

        # model loop
        for day in range(self.sim_length_days):

            if day >= stable_runway:
                treasury_burn = True

            self.iterate_one_day(volume_usd=self.volume_timeseries[day], treasury_burn=treasury_burn)

            if self.rook_price <= 0 or self.treasury_eth <= 0:
                break

            if day < self.sim_length_days - 1:
                self.rook_price_timeseries.append(self.rook_price)
                self.staked_rook_timeseries.append(self.rook_supply.staked)
                self.treasury_rook_timeseries.append(self.rook_supply.treasury)
                self.unclaimed_rook_timeseries.append(self.rook_supply.unclaimed)
                self.burned_rook_timeseries.append(self.rook_supply.burned)
                self.treasury_eth_timeseries.append(self.treasury_eth)

        print(len(self.rook_price_timeseries))
        print(len(self.volume_timeseries[: day + 1]))
        print(len(self.staked_rook_timeseries))
        print(len(self.treasury_rook_timeseries))
        print(len(self.treasury_eth_timeseries))
        print(len(self.unclaimed_rook_timeseries))
        print(len(self.burned_rook_timeseries))

        # construct dataframe
        today = date.today()
        dataframe = pd.DataFrame(
            {
                "date": pd.Series(pd.date_range(today, periods=day + 1, freq="D")),
                "daily_volume": self.volume_timeseries[: day + 1],
                "rook_price": self.rook_price_timeseries,
                "treasury_rook": self.treasury_rook_timeseries,
                "treasury_eth": self.treasury_eth_timeseries,
                "staked_rook": self.staked_rook_timeseries,
                "unclaimed_rook": self.unclaimed_rook_timeseries,
                "burned_rook": self.burned_rook_timeseries,
            }
        )

        return dataframe
