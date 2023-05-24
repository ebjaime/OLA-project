import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from Enviroment import Environment
from Learners import GPTS_Learner


# generic
T = 10                               # horizon of experiment
n_experiments = 30

# pricing
n_prices = 5
prices = np.array([5,6,7,8,9])
p = np.array([0.15,0.1,0.1,0.35,0.1])             # bernoulli distributions for the reward functions
opt_rate = p[np.argmax(p*prices)]                 # optimal arm
print("Pricing (optimal price):")
print("idx: " + str(np.argmax(p*prices)) + "  price: " + str(prices[np.argmax(p*prices)]) + "  rate: " + str(opt_rate)
      + "  price*rate: " + str(opt_rate*prices[np.argmax(p*prices)]))

# advertising
n_bids = 100
min_bid = 0.0
max_bid = 1.0
bids = np.linspace(min_bid, max_bid, n_bids)
sigma_clicks = 10
sigma_costs = 10


def clicks(x):
    return 100 * (1.0 - np.exp(-4*x+3*x**3))


def costs(x):
    return 70 * (1.0 - np.exp(-7*x))


opt_bid = bids[np.argmax(opt_rate*prices[np.argmax(p*prices)]*clicks(bids)-costs(bids))]
fig, ax = plt.subplots()
ax.plot(bids,opt_rate*prices[np.argmax(p*prices)]*clicks(bids),'blue',bids,costs(bids),'orange')
ax.legend(["Number of clicks", "Cumulative Costs"])
ax.axvline(opt_bid,c='red')
print("Advertising (optimal bid):")
print("idx: " + str(np.argmax(opt_rate*prices[np.argmax(p*prices)]*clicks(bids)-costs(bids))) + "  bid: " +
      str(opt_bid) + "  clicks-costs: " + str(clicks(opt_bid)-costs(opt_bid)))

# experiments
gpts_rewards_per_experiment = []

for e in tqdm(range(0, n_experiments)):  # cycle on experiments
    env = Environment(n_arms=n_prices,
                      probabilities=p,
                      bids=bids,
                      average_number_of_clicks=clicks,
                      average_cum_daily_cost=costs,
                      noise_clicks=sigma_clicks,
                      noise_cost=sigma_costs)
    gpts_learner = GPTS_Learner(n_arms=n_bids,
                                arms=bids)

    for t in range(0, T):  # cycle on time horizon

        # GPTS
        pulled_price = np.argmax(p * prices)
        pulled_bid = gpts_learner.pull_arm(opt_rate * prices[pulled_price])
        reward_price, reward_click, reward_cost = env.round(pulled_price, pulled_bid)
        gpts_learner.update(pulled_bid, [reward_click, reward_cost, reward_price])

    gpts_rewards_per_experiment.append(
        prices[np.argmax(p * prices)] * gpts_learner.collected_buyers - gpts_learner.collected_costs)


opt = opt_rate*prices[np.argmax(p*prices)]*clicks(opt_bid) - costs(opt_bid)
plt.figure(0)
plt.ylabel("Cumulative regret")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(opt - gpts_rewards_per_experiment, axis = 0)), 'r')

time = range(0, T)
gpts_std = np.std(np.cumsum(opt - gpts_rewards_per_experiment, axis=1), axis=0)
gpts_metric= np.mean(np.cumsum(opt - gpts_rewards_per_experiment, axis=1), axis=0)
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([gpts_metric - gpts_std, (gpts_metric + gpts_std)[::-1]]),
         alpha=.5, fc='r', ec=None, label='standard deviation')

plt.show()
