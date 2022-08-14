# Modeling Methodology

- Numerical model
- Time steps are days
- Runs for 10 years
- Calculates ROOK price, liquidity, treasury balances, circulating supply amounts, staking APR, at each timestep.

## Price / Liquidity

- All ROOK liquidity modeled as a single, constant-product AMM ROOK/USDC pool a la uni v2. All buy/sell events are simulated in this pool.
  - When buying ROOK with ETH, a lossless ETH-USDC conversion is assumed.
  - In general, buy/sell events are separated in a meaningful order instead of being lumped together at the end of the day. (i.e. keepers buy first, then users claim and sell)
  - At the end of each day, the current pool balances are used for the new market price of ROOK to be used at the beginning of the next time step. Essentially the pool is assumed to be balanced again at the start of each "day". This means the buys and sells likely have more of an effect on price than they would in real life, because the pool would normally be arbed somewhere inbetween. But in this sim, there are no other pools to arb against so that's not reasonable to simulate.
- The liquidity in the ROOK/USDC pool can be modeled in 3 different ways. 2 of them are in essence different ways of indirectly scaling ROOK liquidity with CG volume, and the third is more of an incentivized liquidity program approach.
  - As a % of the total current market cap of ROOK, where mcap = circulating supply \* price. For this option, if the current market cap of ROOK is $100mm, ROOK is currently $100, and the liquidity percentage is 10%, the total value of the ROOK/USDC pool will be $10mm. This means 5mm USDC and $5mm worth of ROOK at $100, or 50k ROOK. This means ROOK liquidity scales with the price and arguably success of the project.
  - As a % of circulating supply of ROOK. This means a portion of the current number of circulating ROOK tokens are assumed to be LP'd in AMMs. So if the current number of circulating tokens is 500k ROOK and the percentage is 10%, 50k ROOK will be matched with an equal $ value of USDC at the beginning of each time step. This means as more and more tokens are temporarily taken out of circulation (as with unclaimed rewards) or permanently taken out of circulation (as with the burn), the number of supplied tokens decreases, and vice versa.
  - As a fixed \$ amount. This means that whatever the current price of ROOK, a fixed $ amount is assumed to be LP'd in AMMS. This model is perhaps more in line with what we'd see from some sort of liquidity mining program.
- At the beginning of each day, the balances of the AMM pool are calculated based on the desired liquidity model, and the relevant balances and market price of ROOK at the end of the previous day. For example, if at the end of the previous day the ROOK price was $100, and the liquidity model was selected to be 10% of market cap:
  - Step 1: Calculate circulating supply as total supply - treasury balance - strat reserves - total burned - total unclaimed (if using ROOK bids). Say circ supply ends up at 600k
  - Step 2: Calculate market cap as ROOK price \* circ supply = $60mm
  - Step 3: Take 10% of mcap and split evenly between ROOK and USDC, so $3mm ROOK (= 30k ROOK tokens) and $3mm USDC.
- Throughout the day, each event modeled in the simulation affects the balances of the pool. These events include:
  - Keepers buying ROOK (if ROOK is the bid token)
  - Users and/or Partners liquidating their rewards (if ROOK is the bid token)
  - Buyback and burn (if ROOK is not the bid token, and the burn is activated)
  - Old stakers unstaking and selling, or new stakers buying and staking
  - The treasury selling to pay expenses (if ROOK is the bid token and the treasury has run out of ETH/stables)

## CG Volume

- 3 ways to model volume growth:
  - Constant (no growth). A fixed daily volume is chosen, and remains unchanged for the entire simulation. Basically a worst-case scenario.
  - Linear Growth. A start volume and maximum achievable daily volume are chosen, and the daily volume grows linearly over the course of the simulation. Unrealistic in practice, but allows for simple projections.
  - [Logistic growth](https://en.wikipedia.org/wiki/Logistic_function). Volume grows slowly at first, then increases exponentially, and finally slows down as it approaches the maximum achievable daily volume. This is a more realistic ideal scenario for growth, where we start slow and steady and then "catch on" and grow rapidly to achieve network effect. Logistic function is often used to model the [Diffusion of Innovations](https://en.wikipedia.org/wiki/Diffusion_of_innovations), which ROOK definitely fits into

## Staking

- Staking is modeled around the concept of a "Target APR". That is, an APR above which people will tend to buy ROOK off the market and stake it to capture the yield, and below which stakers will tend to unstake and sell because better yield is achievable elsewhere. This is an attempt to model buy/sell pressure on the ROOK token due to staking
- APR is modeled as a 3-week moving average (same way it's displayed in the app)
- During the simulation, if the current APR is below the target, 0.1% of the current xROOK supply will unstake and sell per day, until the APR rises to the target. If the APR is below the target, 0.1% of the underlying ROOk value of the current xROOK supply will be bought off market per day and staked until the APR reaches the target.

## Other parameters

- ETH price (held fixed in this model, ETH price is not part of the simulation)
- Early investor nuke
- Treasury runway. Use stablecoins first, when those run out sell ETH, and when that runs out (assuming still negative cash flow to treasury) sell ROOK. Expenses from contributor salaries are modeled based off of preliminary numbers from Comp 3.0. For ROOK grants/options awarded to full time contributors, the estimated $ value of ROOK needed for these grants at current market price at the beginning of each year is allocated from the treasury all at once.
