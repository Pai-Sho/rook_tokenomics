import pandas as pd
import matplotlib.pyplot as plt
from util.params import *
from models.rook_bid import RookBidModel

bid_params = BidDistributionParams(0.1, 0.1, 0.05, 0.05)
protocol_params = ProtocolParams(0.9, bid_params)
ecosystem_params = EcosystemParams(0.001, 0.9, 0.9)
dao_params = DAOParams(25000)
volume_params = VolumeParams(1500000, 0.0035, 250000000)

model = RookBidModel(365*10, protocol_params, bid_params, ecosystem_params, dao_params, volume_model='logistic',
                     volume_params=volume_params, liquidity_model='constant', liquidity_constant=5000000, initial_rook_price=25, treasury_stables=27000000)

df = model.run_sim()

df['daily_volume'].plot()
plt.show()
