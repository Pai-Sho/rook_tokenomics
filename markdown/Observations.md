# Ideal model

- ETH bids
- No staking (except maybe for governance only), no burn
  - Governance becomes scaled down. Relevant stakeholders (Keepers, MMs, Power users, partner protocols, etc) vote
    - Fine tuning protocol params (keeper bid %, greenlight algo)
    - Usage of the treasury.
      - Buyback and burn
      - Handshake deals with new partner protocols
    - Maybe even stuff like the keeper whitelist. New keepers must submit a proposal for review, etc
  - This results in governance being more incentive-aligned. The stakeholders who's best interest it's in for the ecosystem to operate as effectively as possible are the ones most motivated to participate in governance.
- 10% of the bid goes directly to the treasury. The rest by default goes to the user. In the case of a partner protocol, they get to decide what happens to the 90%. They could pass it all through to the user, make a 50/50 split, or keep all of it and distribute that value to their users some other way, like a buyback and burn of their own token, or revenue to their stakers, etc.

# Model where we keep staking

- ETH bids
- No burn
- Default split is 5/20/75 between stakers, treasury and users
- Staking works a bit differently for partners and users:
  - Users:
    - Depending on how much a given user has staked, the bid distribution changes. Say, for each 1 ROOK staked, the user cut of the bid is turned up 0.01%, up to a maximum of 85% if the user has staked 1000 ROOK. This is taken out of the treasury cut.
    - Additionally, Users can opt in the UI to receive their rewards in xROOK instead of ETH. In this case, rather than being able to claim ETH from the coordinator immediately after their order is settled, that ETH is put into a pool with all other user's rewards who've selected this option. At the end of the staking epoch, the ETH is used to buy ROOK off the market (using our own protocol!) and stake it, and the resulting xROOK is distributed to these users pro rata.
    - Staking now has a pseudo "autocompounding" feature, whereby a user can increase their stake by simply routing their tx flow through the CG. Users are incentivized to stake in order to increase their rewards distribution (and their cut of the 5% revenue share), and stakers are incentivized to use the protocol to increase their stake.
  - Partners:
    - As with the other model, if the bid is for an order or tx via a partner protocol, the partner controls the non-treasury and staking portion of the bid.
    - The staking requirement for partners is much higher. Say 10k ROOK to max out the bid distribution as opposed to 1k.
    - Partners can also opt for the autocompound, so they can increase their stake via their users orderflow.
