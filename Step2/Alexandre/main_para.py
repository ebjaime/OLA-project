import matplotlib.pyplot as plt
import numpy as np
import time
from multiprocessing import Pool

from Enviroment import Environment
from Learners import GPTS_Learner


# generic
T = 365                             # horizon of experiment
n_experiments = 10

# pricing
n_prices = 5
prices = np.array([5, 6, 7, 8, 9])
p = np.array([0.15, 0.1, 0.1, 0.35, 0.1])             # bernoulli distributions for the reward functions
opt_rate = p[np.argmax(p*prices)]                 # optimal arm


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


# experiments
gpts_rewards_per_experiment = []


def do_experiment(experience_number):
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

    return prices[np.argmax(p * prices)] * gpts_learner.collected_buyers - gpts_learner.collected_costs


time_beginning = time.time()
if __name__ == '__main__':
    with Pool() as pool:
        for result_experiment in pool.map(do_experiment, range(0, n_experiments)):  # cycle on experiments
            gpts_rewards_per_experiment.append(result_experiment)
        pool.close()
        pool.join()
    time_end = time.time()
    hours = int((time_end - time_beginning) // 3600)
    minutes = int(((time_end - time_beginning) % 3600) // 60)
    seconds = int((time_end - time_beginning) % 60)
    print("Execution time: " + str(hours) + "h " + str(minutes) + "min " + str(seconds) + 's')

    opt = opt_rate*prices[np.argmax(p*prices)]*clicks(opt_bid) - costs(opt_bid)


    plt.figure(0)
    plt.ylabel("Cumulative regret")
    plt.xlabel("t")
    plt.plot(np.cumsum(np.mean(opt - gpts_rewards_per_experiment, axis=0)), 'r')

    time = range(0, T)
    gpts_std = np.std(np.cumsum(opt - gpts_rewards_per_experiment, axis=1), axis=0)
    gpts_metric = np.mean(np.cumsum(opt - gpts_rewards_per_experiment, axis=1), axis=0)
    plt.fill(np.concatenate([time, time[::-1]]),
             np.concatenate([gpts_metric - gpts_std, (gpts_metric + gpts_std)[::-1]]),
             alpha=.5, fc='r', ec=None, label='standard deviation')

    # plt.show()


    plt.figure(1)
    plt.ylabel("Cumulative reward")
    plt.xlabel("t")
    plt.plot(np.cumsum(np.mean(gpts_rewards_per_experiment, axis=0)), 'r')

    time = range(0, T)
    gpts_std = np.std(np.cumsum(gpts_rewards_per_experiment, axis=1), axis=0)
    gpts_metric = np.mean(np.cumsum(gpts_rewards_per_experiment, axis=1), axis=0)
    plt.fill(np.concatenate([time, time[::-1]]),
             np.concatenate([gpts_metric - gpts_std, (gpts_metric + gpts_std)[::-1]]),
             alpha=.5, fc='r', ec=None, label='standard deviation')

    # plt.show()


    plt.figure(2)
    plt.ylabel("Instantaneous regret")
    plt.xlabel("t")
    plt.plot(np.mean(opt - gpts_rewards_per_experiment, axis=0), 'r')

    time = range(0, T)
    gpts_std = np.std(opt - gpts_rewards_per_experiment, axis=0)
    gpts_metric = np.mean(opt - gpts_rewards_per_experiment, axis=0)
    plt.fill(np.concatenate([time, time[::-1]]),
             np.concatenate([gpts_metric - gpts_std, (gpts_metric + gpts_std)[::-1]]),
             alpha=.5, fc='r', ec=None, label='standard deviation')

    # plt.show()


    plt.figure(3)
    plt.ylabel("Instantaneous reward")
    plt.xlabel("t")
    plt.plot(np.mean(gpts_rewards_per_experiment, axis=0), 'r')

    time = range(0, T)
    gpts_std = np.std(gpts_rewards_per_experiment, axis=0)
    gpts_metric = np.mean(gpts_rewards_per_experiment, axis=0)
    plt.fill(np.concatenate([time, time[::-1]]),
             np.concatenate([gpts_metric - gpts_std, (gpts_metric + gpts_std)[::-1]]),
             alpha=.5, fc='r', ec=None, label='standard deviation')

    plt.show()

