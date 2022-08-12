import math
from turtle import st
import numpy as np
import pandas as pd

from datetime import date, datetime

from util.balances import RookSupply, TreasuryEthBalances
from util.params import *


class ModelState:
    def __init__(
        self,
        rook_price: float,
        rook_supply: RookSupply,
        eth_price: float,
        treasury_eth_balance: float,
        treasury_stablecoin_balance: float,
        staking_apr: float,
    ):
        self.rook_price = rook_price
        self.rook_supply = rook_supply
        self.eth_price = eth_price
        self.treasury_eth_balance = treasury_eth_balance
        self.treasury_stablecoin_balance = treasury_stablecoin_balance
        self.staking_apr = staking_apr

    def __str__(self):
        return f"\n\
        ROOK price: {self.rook_price} \n\
        ROOK supply: {self.rook_supply} \n\
        ETH price: {self.eth_price} \n\
        Treasury ETH: {self.treasury_eth_balance} \n\
        "


class CurrentModel:
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
        treasury_stables: float = 26460000,
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
        # rook_supply = RookSupply()
        rook_price = initial_rook_price
        treasury_stables = treasury_stables
        treasury_eth = TreasuryEthBalances().get_treasury_eth()
        eth_price = 2000  # TODO: Model eth price, or accept as input
        staking_apr = 0.003

        # init model states for timeseries
        self.eth_bid_model = ModelState(
            rook_price, RookSupply(), eth_price, treasury_eth, treasury_stables, staking_apr
        )
        self.rook_bid_model = ModelState(
            rook_price, RookSupply(), eth_price, treasury_eth, treasury_stables, staking_apr
        )

        if self.volume_model == "constant":
            self.volume_timeseries = np.full(self.sim_length_days, self.volume_params.start_volume)
        elif self.volume_model == "linear":
            x = np.arange(self.sim_length_days)
            m = (self.volume_params.max_volume - self.volume_params.start_volume) / self.sim_length_days
            y = m * x + self.volume_params.start_volume
            self.volume_timeseries = y
        elif volume_model == "logistic":
            p0 = self.volume_params.start_volume
            k = self.volume_params.max_volume
            r = self.volume_params.volume_growth_rate
            exp = np.exp(r * np.arange(0, sim_length_days))
            self.volume_timeseries = (k * exp * p0) / (k + (exp - 1) * p0)
        else:
            raise ValueError("Unsupported Volume Growth Model")

    def iterate_one_day(self, bid_token: str, volume_usd: float, model_state: ModelState):

        # Set AMM Liquidity:
        if self.liquidity_model == "mcap":
            market_cap = model_state.rook_supply.get_circulating_supply() * model_state.rook_price
            amm_usdc = self.liquidity_constant * market_cap
            amm_rook = amm_usdc / model_state.rook_price
        elif self.liquidity_model == "supply":
            amm_rook = self.liquidity_constant * model_state.rook_supply.get_circulating_supply()
            amm_usdc = amm_rook * model_state.rook_price
        else:
            amm_usdc = self.liquidity_constant / 2
            amm_rook = amm_usdc / model_state.rook_price

        # STEP 1: Keepers buying bid token:
        daily_bid_volume_usd = (
            volume_usd * self.ecosystem_params.mev_volume_ratio * self.protocol_params.target_bid_percent
        )
        if bid_token == "ROOK":
            # Keepers buy ROOK to bid
            keeper_rook_bought = (amm_rook * daily_bid_volume_usd) / (amm_usdc + daily_bid_volume_usd)

            # New AMM pool balances
            amm_rook -= keeper_rook_bought
            amm_usdc += daily_bid_volume_usd
        else:
            # Keepers buy ETH to bid
            keeper_eth_bought = daily_bid_volume_usd / model_state.eth_price

        # STEP 2: Keepers bid:
        if bid_token == "ROOK":
            user_bid = keeper_rook_bought * self.bid_distribution_params.user
            treasury_bid = keeper_rook_bought * self.bid_distribution_params.treasury
            partner_bid = keeper_rook_bought * self.bid_distribution_params.partner
            burn_bid = keeper_rook_bought * self.bid_distribution_params.burn
            stake_bid = keeper_rook_bought * self.bid_distribution_params.stake
        else:
            user_bid = keeper_eth_bought * self.bid_distribution_params.user
            treasury_bid = keeper_eth_bought * self.bid_distribution_params.treasury
            partner_bid = keeper_eth_bought * self.bid_distribution_params.partner
            burn_bid = keeper_eth_bought * self.bid_distribution_params.burn
            stake_bid = keeper_eth_bought * self.bid_distribution_params.stake

        # STEP 3: Users and Partners dumping ROOK if applicable:
        burn_rook_bought = 0
        if bid_token == "ROOK":
            user_rook_sold = (
                user_bid * self.ecosystem_params.user_claim_percent
                + partner_bid * self.ecosystem_params.partner_claim_percent
            )
            user_usdc_bought = (amm_usdc * user_rook_sold) / (amm_rook + user_rook_sold)
            user_rook_unclaimed = user_bid * (1 - self.ecosystem_params.user_claim_percent) + partner_bid * (
                1 - self.ecosystem_params.partner_claim_percent
            )

            # New AMM pool balances
            amm_rook += user_rook_sold
            amm_usdc -= user_usdc_bought
        else:
            burn_rook_bought = burn_bid * (model_state.eth_price / model_state.rook_price)
            burn_usdc_sold = (amm_usdc * burn_rook_bought) / (amm_rook - burn_rook_bought)

            # New AMM pool balances
            amm_rook -= burn_rook_bought
            amm_usdc += burn_usdc_sold

        # STEP 4: Old stakers unstake and sell, or new stakers buy and stake if applicable:
        xrook_underlying_value = model_state.rook_supply.staked / model_state.rook_supply.xrook_total_supply
        staker_rook_sold = 0
        staker_rook_bought = 0
        xrook_minted = 0
        xrook_burned = 0

        if bid_token == "ROOK":

            if model_state.staking_apr < self.ecosystem_params.target_staking_apr:
                xrook_burned = model_state.rook_supply.xrook_total_supply * 0.001

                rook_unstaked = xrook_burned * xrook_underlying_value
                staker_rook_sold = rook_unstaked
                staker_usdc_bought = (amm_usdc * staker_rook_sold) / (amm_rook + staker_rook_sold)

                amm_rook += staker_rook_sold
                amm_usdc -= staker_usdc_bought

            elif model_state.staking_apr > self.ecosystem_params.target_staking_apr:
                rook_staked = model_state.rook_supply.staked * 0.001
                xrook_minted = rook_staked * xrook_underlying_value

                staker_usdc_sold = rook_staked * model_state.rook_price
                staker_rook_bought = (amm_rook * staker_usdc_sold) / (amm_usdc + staker_usdc_sold)

                amm_rook -= staker_rook_bought
                amm_usdc += staker_usdc_sold

        # STEP 5: Treasury sells ETH when out of stables, and ROOK when out of ETH
        treasury_usdc_bought = 0
        treasury_rook_sold = 0
        treasury_eth_sold = 0
        treasury_eth_usd = model_state.treasury_eth_balance * model_state.eth_price
        if model_state.treasury_stablecoin_balance >= self.dao_params.daily_treasury_burn_usd:
            model_state.treasury_stablecoin_balance -= self.dao_params.daily_treasury_burn_usd
        elif treasury_eth_usd >= self.dao_params.daily_treasury_burn_usd:
            treasury_eth_sold = self.dao_params.daily_treasury_burn_usd / model_state.eth_price
        elif treasury_eth_usd < self.dao_params.daily_treasury_burn_usd:
            treasury_usdc_bought = self.dao_params.daily_treasury_burn_usd
            treasury_rook_sold = (amm_rook * treasury_usdc_bought) / (amm_usdc - treasury_usdc_bought)

        # New AMM pool balances
        amm_rook += treasury_rook_sold
        amm_usdc -= treasury_usdc_bought

        # Update ROOK price and supply balances
        model_state.rook_price = amm_usdc / amm_rook

        if bid_token == "ROOK":
            model_state.rook_supply.staked += stake_bid - staker_rook_sold + staker_rook_bought
            model_state.rook_supply.treasury += (
                treasury_bid - treasury_rook_sold
            )  # - self.dao_params.daily_treasury_burn_rook
            model_state.rook_supply.unclaimed += user_rook_unclaimed
            model_state.rook_supply.burned += burn_bid
            model_state.rook_supply.xrook_total_supply += xrook_minted - xrook_burned
            model_state.treasury_eth_balance -= treasury_eth_sold
            model_state.staking_apr = (((stake_bid / model_state.rook_supply.staked) * 365) * (1 / 21)) + (
                model_state.staking_apr * 20 / 21
            )
        else:
            model_state.rook_supply.treasury -= treasury_rook_sold  # + self.dao_params.daily_treasury_burn_rook
            model_state.treasury_eth_balance += treasury_bid - treasury_eth_sold
            model_state.rook_supply.burned += burn_rook_bought

    def run_sim_rook(self):

        # ROOK bid model
        rook_price_timeseries = [self.rook_bid_model.rook_price]
        staked_rook_timeseries = [self.rook_bid_model.rook_supply.staked]
        treasury_rook_timeseries = [self.rook_bid_model.rook_supply.treasury]
        unclaimed_rook_timeseries = [self.rook_bid_model.rook_supply.unclaimed]
        burned_rook_timeseries = [self.rook_bid_model.rook_supply.burned]
        treasury_eth_timeseries = [self.rook_bid_model.treasury_eth_balance]
        staking_apr_timeseries = [self.rook_bid_model.staking_apr]
        treasury_stables_timeseries = [self.rook_bid_model.treasury_stablecoin_balance]

        # model loop
        for day in range(self.sim_length_days):

            # At the beginning of each year, allocate the appropriate amount of ROOK for contributor options
            if day % 365 == 0:
                rook_options_amount = self.dao_params.yearly_treasury_burn_rook / self.rook_bid_model.rook_price
                self.rook_bid_model.rook_supply.treasury -= rook_options_amount

            self.iterate_one_day(
                bid_token="ROOK",
                volume_usd=self.volume_timeseries[day],
                model_state=self.rook_bid_model,
            )

            # Stop the simulation if ROOK goes to 0, or the treasury runs out of ROOK or ETH
            if (
                self.rook_bid_model.rook_price <= 0
                or self.rook_bid_model.rook_supply.treasury <= 0
                or self.rook_bid_model.treasury_eth_balance <= 0
            ):
                break

            if day < self.sim_length_days - 1:
                rook_price_timeseries.append(self.rook_bid_model.rook_price)
                staked_rook_timeseries.append(self.rook_bid_model.rook_supply.staked)
                treasury_rook_timeseries.append(self.rook_bid_model.rook_supply.treasury)
                unclaimed_rook_timeseries.append(self.rook_bid_model.rook_supply.unclaimed)
                burned_rook_timeseries.append(self.rook_bid_model.rook_supply.burned)
                treasury_eth_timeseries.append(self.rook_bid_model.treasury_eth_balance)
                staking_apr_timeseries.append(self.rook_bid_model.staking_apr)
                treasury_stables_timeseries.append(self.rook_bid_model.treasury_stablecoin_balance)

        # construct dataframe
        today = date.today()
        dataframe = pd.DataFrame(
            {
                "date": pd.Series(pd.date_range(today, periods=day + 1, freq="D")),
                "daily_volume": self.volume_timeseries[: day + 1],
                "rook_price": rook_price_timeseries,
                "treasury_rook": treasury_rook_timeseries,
                "treasury_eth": treasury_eth_timeseries,
                "treasury_stables": treasury_stables_timeseries,
                "staked_rook": staked_rook_timeseries,
                "unclaimed_rook": unclaimed_rook_timeseries,
                "burned_rook": burned_rook_timeseries,
                "staking_apr": staking_apr_timeseries,
            }
        )

        return dataframe

    def run_sim_eth(self):

        # ETH bid model
        rook_price_timeseries = [self.eth_bid_model.rook_price]
        staked_rook_timeseries = [self.eth_bid_model.rook_supply.staked]
        treasury_rook_timeseries = [self.eth_bid_model.rook_supply.treasury]
        unclaimed_rook_timeseries = [self.eth_bid_model.rook_supply.unclaimed]
        burned_rook_timeseries = [self.eth_bid_model.rook_supply.burned]
        treasury_eth_timeseries = [self.eth_bid_model.treasury_eth_balance]
        staking_apr_timeseries = [self.eth_bid_model.staking_apr]
        treasury_stables_timeseries = [self.eth_bid_model.treasury_stablecoin_balance]

        print(rook_price_timeseries)
        print(staked_rook_timeseries)
        print(staking_apr_timeseries)
        print(unclaimed_rook_timeseries)

        # model loop
        for day in range(self.sim_length_days):

            self.iterate_one_day(
                bid_token="ETH",
                volume_usd=self.volume_timeseries[day],
                model_state=self.eth_bid_model,
            )

            if (
                self.eth_bid_model.rook_price <= 0
                or self.eth_bid_model.rook_supply.treasury <= 0
                or self.eth_bid_model.treasury_eth_balance <= 0
            ):
                break

            if day < self.sim_length_days - 1:
                rook_price_timeseries.append(self.eth_bid_model.rook_price)
                staked_rook_timeseries.append(self.eth_bid_model.rook_supply.staked)
                treasury_rook_timeseries.append(self.eth_bid_model.rook_supply.treasury)
                unclaimed_rook_timeseries.append(self.eth_bid_model.rook_supply.unclaimed)
                burned_rook_timeseries.append(self.eth_bid_model.rook_supply.burned)
                treasury_eth_timeseries.append(self.eth_bid_model.treasury_eth_balance)
                staking_apr_timeseries.append(self.eth_bid_model.staking_apr)
                treasury_stables_timeseries.append(self.eth_bid_model.treasury_stablecoin_balance)

        # construct dataframe
        today = date.today()
        dataframe = pd.DataFrame(
            {
                "date": pd.Series(pd.date_range(today, periods=day + 1, freq="D")),
                "daily_volume": self.volume_timeseries[: day + 1],
                "rook_price": rook_price_timeseries,
                "treasury_rook": treasury_rook_timeseries,
                "treasury_stables": treasury_stables_timeseries,
                "treasury_eth": treasury_eth_timeseries,
                "staked_rook": staked_rook_timeseries,
                "unclaimed_rook": unclaimed_rook_timeseries,
                "burned_rook": burned_rook_timeseries,
                "staking_apr": staking_apr_timeseries,
            }
        )

        return dataframe
