import numpy as np

from Environment import Environment
from Learners import TS_Pricing_Learner, UCB_Pricing_Learner


## Simulations

import matplotlib.pyplot as plt
from tqdm import tqdm

# generic
T = 365  # horizon of experiment
n_experiments = 3000  # since the reward functions are stochastic, to better visualize the results and remove the noise we do multiple experiments

# pricing
n_prices = 5
prices = np.array([5, 6, 7, 8, 9])
p = np.array([0.15, 0.1, 0.1, 0.35, 0.1])  # bernoulli distributions for the reward functions
opt_rate = p[np.argmax(p * prices)]  # optimal arm
print("Pricing (optimal price):")
print("idx: " + str(np.argmax(p * prices)) + "  price: " + str(prices[np.argmax(p * prices)]) + "  rate: " + str(
    opt_rate) + "  price*rate: " + str(opt_rate * prices[np.argmax(p * prices)]))

# advertising
n_bids = 100
min_bid = 0.0
max_bid = 1.0
bids = np.linspace(min_bid, max_bid, n_bids)
sigma_clicks = 10
sigma_costs = 10


def clicks(x):
    return 100 * (1.0 - np.exp(-4 * x + 3 * x ** 3))


def costs(x):
    return 70 * (1.0 - np.exp(-7 * x))


def profit(x, value_of_click):
    return value_of_click * clicks(x) - costs(x)


opt_bid = bids[np.argmax(profit(bids, opt_rate * prices[np.argmax(p * prices)]))]
fig, ax = plt.subplots()
ax.plot(bids, clicks(bids) * opt_rate, 'blue', bids, costs(bids), 'orange')
ax.legend(["Number of clicks", "Cumulative Costs"])
ax.axvline(opt_bid, c='red')
plt.show()
print(" ")
print("Advertising (optimal bid):")
print("idx: " + str(np.argmax(profit(bids, opt_rate * prices[np.argmax(p * prices)]))) + "  bid: " + str(opt_bid) + "  clicks-costs: " + str(
    profit(opt_bid, opt_rate * prices[np.argmax(p * prices)])))

## Known advertising curves (1)

# experiments
ts_rewards_per_experiment = []
ucb_rewards_per_experiment = []

for e in tqdm(range(0, n_experiments)):  # cycle on experiments

    env = Environment(n_arms=n_prices,
                      probabilities=p,
                      bids=bids,
                      average_number_of_clicks=clicks,
                      average_cum_daily_cost=costs,
                      noise_clicks=sigma_clicks,
                      noise_cost=sigma_costs)
    ts_learner = TS_Pricing_Learner(n_arms=n_prices,
                                    prices=prices)
    ucb_learner = UCB_Pricing_Learner(n_arms=n_prices,
                                      prices=prices)

    ts_price_history = np.array([])
    ts_collected_clicks = np.array([])
    ts_collected_buyers = np.array([])
    ts_collected_costs = np.array([])
    ucb_price_history = np.array([])
    ucb_collected_clicks = np.array([])
    ucb_collected_buyers = np.array([])
    ucb_collected_costs = np.array([])

    for t in range(0, T):  # cycle on time horizon

        # TS
        pulled_price, sampled_value_of_click = ts_learner.pull_arm()
        pulled_bid = np.argmax(profit(bids, sampled_value_of_click))
        reward_price, reward_click, reward_cost = env.round(pulled_price, pulled_bid)
        ts_learner.update(pulled_price, reward_price, reward_click)

        ts_price_history = np.append(ts_price_history, prices[pulled_price])
        ts_collected_clicks = np.append(ts_collected_clicks, reward_click)
        ts_collected_buyers = np.append(ts_collected_buyers, reward_price)
        ts_collected_costs = np.append(ts_collected_costs, reward_cost)

        # UCB
        pulled_price, upper_value_of_click = ucb_learner.pull_arm()
        pulled_bid = np.argmax(profit(bids, upper_value_of_click))
        reward_price, reward_click, reward_cost = env.round(pulled_price, pulled_bid)
        ucb_learner.update(pulled_price, reward_price, reward_click)

        ucb_price_history = np.append(ucb_price_history, prices[pulled_price])
        ucb_collected_clicks = np.append(ucb_collected_clicks, reward_click)
        ucb_collected_buyers = np.append(ucb_collected_buyers, reward_price)
        ucb_collected_costs = np.append(ucb_collected_costs, reward_cost)

    # ts_rewards_per_experiment.append(ts_learner.actual_reward * ts_collected_clicks - ts_collected_costs)
    ts_rewards_per_experiment.append(ts_price_history * ts_collected_buyers - ts_collected_costs)
    # ucb_rewards_per_experiment.append(ucb_learner.actual_reward * ucb_collected_clicks - ucb_collected_costs)
    ucb_rewards_per_experiment.append(ucb_price_history * ucb_collected_buyers - ucb_collected_costs)

opt = opt_rate * prices[np.argmax(p * prices)] * clicks(opt_bid) - costs(opt_bid)

plt.figure(0)
plt.ylabel("Cumulative regret")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - ucb_rewards_per_experiment, axis=0)), 'b')

time = range(0, T)
ts_std = np.std(np.cumsum(opt - ts_rewards_per_experiment, axis=1), axis=0)
ts_metric= np.mean(np.cumsum(opt - ts_rewards_per_experiment, axis=1), axis=0)
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([ts_metric - ts_std, (ts_metric + ts_std)[::-1]]),
         alpha=.5, fc='r', ec=None, label='standard deviation')

ucb_std = np.std(np.cumsum(opt - ucb_rewards_per_experiment, axis=1), axis=0)
ucb_metric = np.mean(np.cumsum(opt - ucb_rewards_per_experiment, axis=1), axis=0)
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([ucb_metric - ucb_std, (ucb_metric+ ucb_std)[::-1]]),
         alpha=.5, fc='b', ec=None, label='standard deviation')

plt.legend(["TS", "UCB"])
# plt.show()


plt.figure(1)
plt.ylabel("Cumulative reward")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(ts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(ucb_rewards_per_experiment, axis=0)), 'b')

time = range(0, T)
ts_std = np.std(np.cumsum(ts_rewards_per_experiment, axis=1), axis=0)
ts_metric= np.mean(np.cumsum(ts_rewards_per_experiment, axis=1), axis=0)
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([ts_metric - ts_std, (ts_metric + ts_std)[::-1]]),
         alpha=.5, fc='r', ec=None, label='standard deviation')

ucb_std = np.std(np.cumsum(ucb_rewards_per_experiment, axis=1), axis=0)
ucb_metric = np.mean(np.cumsum(ucb_rewards_per_experiment, axis=1), axis=0)
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([ucb_metric - ucb_std, (ucb_metric+ ucb_std)[::-1]]),
         alpha=.5, fc='b', ec=None, label='standard deviation')

plt.legend(["TS", "UCB"])
# plt.show()


plt.figure(2)
plt.ylabel("Instantaneous regret")
plt.xlabel("t")
plt.plot(np.mean(opt - ts_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(opt - ucb_rewards_per_experiment, axis=0), 'b')

time = range(0, T)
ts_std = np.std(opt - ts_rewards_per_experiment, axis=0)
ts_metric= np.mean(opt - ts_rewards_per_experiment, axis=0)
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([ts_metric - ts_std, (ts_metric + ts_std)[::-1]]),
         alpha=.5, fc='r', ec=None, label='standard deviation')

ucb_std = np.std(opt - ucb_rewards_per_experiment, axis=0)
ucb_metric = np.mean(opt - ucb_rewards_per_experiment, axis=0)
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([ucb_metric - ucb_std, (ucb_metric+ ucb_std)[::-1]]),
         alpha=.5, fc='b', ec=None, label='standard deviation')

plt.legend(["TS", "UCB"])
# plt.show()


plt.figure(3)
plt.ylabel("Instantaneous reward")
plt.xlabel("t")
plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(ucb_rewards_per_experiment, axis=0), 'b')

time = range(0, T)
ts_std = np.std(ts_rewards_per_experiment, axis=0)
ts_metric= np.mean(ts_rewards_per_experiment, axis=0)
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([ts_metric - ts_std, (ts_metric + ts_std)[::-1]]),
         alpha=.5, fc='r', ec=None, label='standard deviation')

ucb_std = np.std(ucb_rewards_per_experiment, axis=0)
ucb_metric = np.mean(ucb_rewards_per_experiment, axis=0)
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([ucb_metric - ucb_std, (ucb_metric+ ucb_std)[::-1]]),
         alpha=.5, fc='b', ec=None, label='standard deviation')

plt.legend(["TS", "UCB"])
plt.show()
