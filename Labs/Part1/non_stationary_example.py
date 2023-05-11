import numpy as np
import matplotlib.pyplot as plt

from Non_Stationary_Environment import Non_Stationary_Environment
from TS_Learner import TS_Learner
from SWTS_Learner import SWTS_Learner

n_arms = 4
n_phases = 4
p = np.array([[.15, .1, .2, .35],
              [.45, .21, .2, .35],
              [.1, .1, .5, .15],
              [.1, .21, .1, .15]])

T = 500
phases_len = int(T/n_phases)
n_experiments = 100
ts_rewards_per_experiment = []
swts_rewards_per_experiment = []
window_size = int(T**0.5)

if __name__ == '__main__':
    for e in range(n_experiments):
        ts_env = Non_Stationary_Environment(n_arms, p, T)
        ts_learner = TS_Learner(n_arms)
        
        swts_env = Non_Stationary_Environment(n_arms, p, T)
        swts_learner = SWTS_Learner(n_arms, window_size=window_size)

        for t in range(T):
            pulled_arm = ts_learner.pull_arm()
            reward = ts_env.round(pulled_arm)
            ts_learner.update(pulled_arm, reward)

            pulled_arm = swts_learner.pull_arm()
            reward = swts_env.round(pulled_arm)
            swts_learner.update(pulled_arm, reward)
            
        ts_rewards_per_experiment.append(ts_learner.collected_rewards)
        swts_rewards_per_experiment.append(swts_learner.collected_rewards)
    
    ts_instantaneous_regret = np.zeros(T)
    swts_instantaneous_regret = np.zeros(T)
    opt_per_phases = p.max(axis=1)
    optimum_per_round = np.zeros(T)

    for i in range(n_phases):
        t_index = range(i*phases_len, (i+1)*phases_len)
        optimum_per_round[t_index] = opt_per_phases[i]
        ts_instantaneous_regret[t_index] = opt_per_phases[i] - np.mean(ts_rewards_per_experiment, axis=0)[t_index]
        swts_instantaneous_regret[t_index] = opt_per_phases[i] - np.mean(swts_rewards_per_experiment, axis=0)[t_index]
    
    plt.figure(0)
    plt.xlabel('t')
    plt.ylabel('Reward')
    plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
    plt.plot(np.mean(swts_rewards_per_experiment, axis=0), 'b')
    plt.plot(optimum_per_round, 'k--')
    plt.legend(["TS", "SW-TS", "Optimum"])
    plt.show()    

    plt.figure(1)
    plt.xlabel('t')
    plt.ylabel('Regret')
    plt.plot(np.cumsum(ts_instantaneous_regret), 'r')
    plt.plot(np.cumsum(swts_instantaneous_regret), 'b')
    plt.legend(["TS", "SW-TS"])
    plt.show()
    
        