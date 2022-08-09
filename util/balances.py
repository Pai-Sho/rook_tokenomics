import os
from unicodedata import decimal
import requests
import json
from dotenv import load_dotenv

from web3 import Web3

from util.addresses import addresses
from util.contracts import fetch_abi

load_dotenv()
INFURA_KEY = os.getenv("INFURA_KEY")

WEB3 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/{}".format(INFURA_KEY)))


class RookSupply:
    def __init__(self):
        self.web3 = WEB3
        rook_address = addresses.rook
        rook_abi = fetch_abi(rook_address)
        rook_contract = self.web3.eth.contract(address=rook_address, abi=rook_abi)
        decimals = 10 ** rook_contract.functions.decimals().call()

        # Fetch ROOK balances
        total_supply = rook_contract.functions.totalSupply().call()
        treasury = rook_contract.functions.balanceOf(addresses.treasury).call()
        strategic_reserves = rook_contract.functions.balanceOf(addresses.strategic_reserves).call()
        staked = rook_contract.functions.balanceOf(addresses.liquidity_pool_v4).call()

        # Set initial ROOK balances
        self.total_supply = total_supply / decimals
        self.treasury = treasury / decimals
        self.strategic_reserves = strategic_reserves / decimals
        self.staked = staked / decimals
        self.burned = 0
        self.unclaimed = 0

    def get_circulating_supply(self):
        return self.total_supply - self.treasury - self.strategic_reserves - self.burned - self.unclaimed


class TreasuryEthBalances:
    def __init__(self):
        self.web3 = WEB3

        weth_abi = fetch_abi(addresses.weth)
        lpv4_abi = fetch_abi(addresses.liquidity_pool_v4)
        kweth_abi = fetch_abi(addresses.kweth)

        weth_contract = self.web3.eth.contract(address=addresses.weth, abi=weth_abi)
        lpv4_contract = self.web3.eth.contract(address=addresses.liquidity_pool_v4, abi=lpv4_abi)
        kweth_contract = self.web3.eth.contract(address=addresses.kweth, abi=kweth_abi)

        weth_decimals = 10 ** weth_contract.functions.decimals().call()
        kweth_decimals = 10 ** kweth_contract.functions.decimals().call()

        # Fetch Treasury balances
        eth_balance = self.web3.eth.get_balance(addresses.treasury)
        weth_balance = weth_contract.functions.balanceOf(addresses.treasury).call()
        kweth_balance = kweth_contract.functions.balanceOf(addresses.treasury).call()

        # Get kweth underlying conversion
        kweth_to_underlying = (
            weth_contract.functions.balanceOf(addresses.liquidity_pool_v4).call()
            / kweth_contract.functions.totalSupply().call()
        )

        self.eth_balance = eth_balance / 10**18
        self.weth_balance = weth_balance / weth_decimals
        self.kweth_underlying_balance = (kweth_balance / kweth_decimals) * kweth_to_underlying

    def get_treasury_eth(self):
        return self.eth_balance + self.weth_balance + self.kweth_underlying_balance
