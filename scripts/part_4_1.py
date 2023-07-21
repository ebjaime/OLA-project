"""
## Environment
"""

import numpy as np

class Environment_Multiple_Contexts():

  def __init__(self, n_arms, probabilities, bids, average_number_of_clicks, average_cum_daily_cost,noise_clicks,noise_cost):
    self.n_arms = n_arms                                            # number of prices
    self.probabilities = probabilities                              # conversion rates for every price/arm and for every class (size: |classes|x|arms|)
    self.bids = bids                                                # bids
    self.average_number_of_clicks = average_number_of_clicks        # curves of average number of clicks(y = f(bids)), one for each class
    self.average_cum_daily_cost = average_cum_daily_cost            # curves of cumulative daily cost (y = g(bids)), one for each class
    self.noise_clicks = noise_clicks                                # gaussian noises for the average number of clicks sampling, one for each class
    self.noise_cost = noise_cost                                    # gaussian noises for the cumulative daily cost sampling, one for each class

  # daily rewards (given a class "i")
  def bidding_round(self, pulled_bid, i):
    clicks = int(np.random.normal(self.average_number_of_clicks[i](self.bids[pulled_bid]),self.noise_clicks[i]))        # number of people that click on the ad
    reward_click = clicks if clicks >= 0 else 0
    costs = np.random.normal(self.average_cum_daily_cost[i](self.bids[pulled_bid]),self.noise_cost[i])                  # cumulative daily cost
    reward_cost = costs if costs > 0 else 1
    return reward_click, reward_cost

  # pricing rewards (given a class "i")
  def pricing_round(self, pulled_price, i):
    reward_price = np.random.binomial(1,self.probabilities[i][pulled_price])                         # number of people that buy once they clicked
    return reward_price

"""## Pricing"""

class Learner:
  def __init__(self,n_arms):
    self.n_arms = n_arms
    self.t = 0                                              # current round value
    self.rewards_per_arm = x = [[] for i in range(n_arms)]  # value of collected rewards for each round and for each arm
    self.collected_rewards = np.array([])                   # values of collected rewards for each round

  # function that updates the observation's list once the reward is returned by the environment
  def update_observations(self, pulled_arm, reward):
    self.rewards_per_arm[pulled_arm].append(reward)
    self.collected_rewards = np.append(self.collected_rewards,reward)

class TS_Pricing_Learner(Learner): # Thompson-Sampling (reward: number of conversions; actual_reward:  price*conversion_rate)
  def __init__(self,n_arms,prices):
    super().__init__(n_arms)                    # number of prices
    self.beta_parameters = np.ones((n_arms,2))  # parameters of beta distributions
    self.prices = prices                        # prices (array)

  def pull_arm(self):
    if(self.t < self.n_arms):
      return self.t
    sampled = np.random.beta(self.beta_parameters[:,0],self.beta_parameters[:,1])*self.prices
    idx = np.argmax(sampled)
    return idx

  # update parameters each time a reward in {0,1} is observed
  def update(self,pulled_arm, reward):
    self.t += 1
    self.update_observations(pulled_arm,reward*self.prices[pulled_arm])
    self.beta_parameters[pulled_arm,0] = self.beta_parameters[pulled_arm,0] + reward
    self.beta_parameters[pulled_arm,1] = self.beta_parameters[pulled_arm,1] + 1 - reward

class UCB_Pricing_Learner(Learner): # UCB1 (reward: number of conversions; actual_reward:  price*conversion_rate)
  def __init__(self,n_arms,prices):
    super().__init__(n_arms)                              # number of arms/prices
    self.empirical_means = np.zeros(n_arms)               # mean reward for each arm (conversion rate)
    self.confidence = np.zeros(n_arms)                    # confidence bound for each arm
    self.prices = prices                                  # prices (array)

  def pull_arm(self):
    if(self.t < self.n_arms):
      return self.t
    upper_bound = self.empirical_means + self.confidence
    pulled_arm = np.random.choice(np.where(upper_bound == upper_bound.max())[0])
    return pulled_arm

  def update(self, pulled_arm, reward):
    self.t += 1
    self.update_observations(pulled_arm, reward*self.prices[pulled_arm])
    self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm]*(len(self.rewards_per_arm[pulled_arm]) - 1) + reward*self.prices[pulled_arm] ) / len(self.rewards_per_arm[pulled_arm])
    self.confidence[pulled_arm] = self.prices[pulled_arm]*np.sqrt(2*np.log(self.t)/len(self.rewards_per_arm[pulled_arm]))

"""## Advertising"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class GPTS_Learner(Learner):
  # constructor takes as input the number arms (bids), intializes to 0 their means and to an aye their standard deviation; we also need to store the arm that we pull at each round to fot the GPs
  def __init__(self, n_arms, arms, alpha_clicks, alpha_costs, normalize_y = False, step = 1, step_delay = 366, n_restarts_optimizer = 0, plot = False):
    super().__init__(n_arms)
    self.arms = arms
    self.means_clicks = np.ones(self.n_arms) * 1e3
    self.means_costs = np.ones(self.n_arms) * 1e3
    self.sigmas_clicks = np.ones(self.n_arms)
    self.sigmas_costs = np.ones(self.n_arms)
    self.pulled_arms = []
    self.collected_clicks = np.array([])
    self.collected_costs = np.array([])
    # we initialize the parameters of the kernel and the two GPs
    if normalize_y:
      kernel_clicks = C(2000.,(1000.,5000.)) * RBF(0.3,(0.01,0.6)) # for normalized
      kernel_costs = C(2000.,(1000.,5000.)) * RBF(0.3,(0.01,0.6))  # for normalized
    else:
      kernel_clicks = C(1,(1e3,1e4)) * RBF(1,(0.1,1))
      kernel_costs = C(1, (1e3,1e4)) * RBF(1,(0.1,1))
    self.gp_clicks = GaussianProcessRegressor(kernel = kernel_clicks, alpha = alpha_clicks**2, normalize_y = normalize_y, n_restarts_optimizer = n_restarts_optimizer)
    self.gp_costs = GaussianProcessRegressor(kernel = kernel_costs, alpha = alpha_costs**2, normalize_y = normalize_y, n_restarts_optimizer = n_restarts_optimizer)
    self.plot = plot
    self.step = step
    self.step_delay = step_delay

  # we also need to update the value of the least pulled arm (reward[0]: n_clicks, reward[1]: costs)
  def update_observations(self, arm_idx, reward):
    super().update_observations(arm_idx, reward)
    self.pulled_arms.append(self.arms[arm_idx])

  # update the GP estimations and consequently the means and sigmas of each arm
  def update_model(self):
    # trining inputs and targets
    if self.t < self.step_delay or not(self.t % self.step):
      x = np.atleast_2d(self.pulled_arms).T
      y_clicks = self.collected_clicks
      y_costs = self.collected_costs
      # fit the GP
      if len(y_clicks) > 1:
        self.gp_clicks.fit(x,y_clicks)
        self.gp_costs.fit(x,y_costs)
        # update values of means and sigmas with the new predictions
        self.means_clicks, self.sigmas_clicks = self.gp_clicks.predict(np.atleast_2d(self.arms).T, return_std = True)
        self.means_costs, self.sigmas_costs = self.gp_costs.predict(np.atleast_2d(self.arms).T, return_std = True)
        self.sigmas_clicks = np.maximum(self.sigmas_clicks, 1e-2)
        self.sigmas_costs = np.maximum(self.sigmas_costs, 1e-2)

      if self.plot:
        plt.figure()
        plt.title("Clicks:" + str(self.t))
        plt.plot(self.arms, clicks(self.arms), 'r:', label = r'$n(x)$')
        plt.plot(x.ravel(), y_clicks.ravel(), 'ro', label = u'Observed Clicks')
        plt.plot(self.arms, self.means_clicks, 'b-', label = u'Predicted Clicks')
        plt.fill(np.concatenate([self.arms,self.arms[::-1]]),
                np.concatenate([self.means_clicks - 1.96 * self.sigmas_clicks , (self.means_clicks + 1.96 * self.sigmas_clicks)[::-1]]),
                alpha = .5, fc = 'b', ec = 'None', label = '95% conf interval')
        plt.xlabel('$x')
        plt.ylabel('$n(x)$')
        plt.legend(loc = 'lower right')
        plt.show()

        plt.figure()
        plt.title("Costs:" + str(self.t))
        plt.plot(self.arms, costs(self.arms), 'r:', label = r'$c(x)$')
        plt.plot(x.ravel(), y_costs.ravel(), 'ro', label = u'Observed Costs')
        plt.plot(self.arms, self.means_costs, 'b-', label = u'Predicted Costs')
        plt.fill(np.concatenate([self.arms,self.arms[::-1]]),
                np.concatenate([self.means_costs - 1.96 * self.sigmas_costs , (self.means_costs + 1.96 * self.sigmas_costs)[::-1]]),
                alpha = .5, fc = 'b', ec = 'None', label = '95% conf interval')
        plt.xlabel('$x')
        plt.ylabel('$c(x)$')
        plt.legend(loc = 'lower right')
        plt.show()

  # functions that calls the functions implemented above (reward[0]: n_clicks, reward[1]: costs)
  def update(self, pulled_arm, reward):
    self.t += 1
    self.update_observations(pulled_arm, reward)
    self.collected_clicks = np.append(self.collected_clicks,reward[0])
    self.collected_costs = np.append(self.collected_costs,reward[1])
    self.update_model()

  # function in which the learner chooses the arm to pull at each round
  def pull_arm(self):
    # returns index of the maximum value drawn from the arm normal distribution
    samples = np.random.normal(self.means_clicks - self.means_costs, np.sqrt(self.sigmas_clicks**2 + self.sigmas_costs**2))
    return np.argmax(samples)

class GPUCB_Learner(GPTS_Learner):
  def __init__(self, n_arms, arms, alpha_clicks, alpha_costs, normalize_y = False, step = 1, step_delay = 366, n_restarts_optimizer = 0, plot = False):
    super().__init__(n_arms, arms, alpha_clicks, alpha_costs, normalize_y, step, step_delay, n_restarts_optimizer, plot)

  # returns index of the maximum UCB from the arm normal distribution (coefficient for CI of order 1-1/T: 3.0)
  def pull_arm(self):
    upper_bound = self.means_clicks - self.means_costs + 3.0*np.sqrt(self.sigmas_clicks**2 + self.sigmas_costs**2)
    pulled_arm = np.random.choice(np.where(upper_bound == upper_bound.max())[0])
    return pulled_arm

"""## Simulation"""

import matplotlib.pyplot as plt
from tqdm import tqdm

# generic
T = 365                                # horizon of experiment
n_experiments = 100                    # since the reward functions are stochastic, to better visualize the results and remove the noise we do multiple experiments

# pricing
n_prices = 5
prices = [5,6,7,8,9]
p = np.array([[0.36,0.3,0.257,0.313,0.2],
              [0.5,0.3,0.257,0.225,0.2],
              [0.36,0.3,0.257,0.225,0.278]])           # bernoulli distributions for the reward functions
opt_rate = [p[0][np.argmax(p[0]*prices)],
            p[1][np.argmax(p[1]*prices)],
            p[2][np.argmax(p[2]*prices)]]                 # optimal arm

for i in range(0,3):
    print("Pricing (optimal price) context ", str(i))
    print("idx: " + str(np.argmax(p[i]*prices)) + "  price: " + str(prices[np.argmax(p[i]*prices)]) + "  rate: " + str(opt_rate[i]) + "  price*rate: " + str(opt_rate[i]*prices[np.argmax(p[i]*prices)]))

# advertising
n_bids = 100
min_bid = 0.0
max_bid = 1.0
bids = np.linspace(min_bid, max_bid, n_bids)
sigma_clicks = [3, 2.5, 3]
sigma_costs = [3, 2.5, 3]

params_clicks = [[100, 4, 3, 3],
                 [95,  6, 5, 3],
                 [100, 5, 4, 2]]
params_costs = [[70, 7],
                [80, 6],
                [50, 4]]
def clicks(x, params=[100, 4, 3, 3]):
  return params[0] * (1.0 - np.exp(-params[1]*x+params[2]*x**params[3]))

def costs(x, params=[70, 7]):
  return params[0] * (1.0 - np.exp(-params[1]*x))


opt_bid = np.zeros(3)
for i in range(0,3):
    opt_bid[i] = bids[np.argmax(opt_rate[i]*prices[np.argmax(p[i]*prices)]*clicks(bids, params_clicks[i])-costs(bids, params_costs[i]))]

fig, ax = plt.subplots(3, figsize=(8,10))
for i in range(0, 3):
    ax[i].plot(bids,clicks(bids, params_clicks[i]),'blue',bids, costs(bids, params_costs[i]),'orange')
    ax[i].legend(["Number of clicks", "Cumulative Costs"])
    ax[i].axvline(opt_bid[i],c='red')
    print("Advertising (optimal bid):")
    print("idx: " + str(np.argmax(opt_rate[i]*prices[np.argmax(p[i]*prices)]*clicks(bids, params_clicks[i])-costs(bids, params_costs[i]))) + "  bid: " + str(opt_bid[i]) + "  clicks-costs: " + str(clicks(opt_bid[i], params_clicks[i])-costs(opt_bid[i], params_costs[i])))

# Function wrapper to be able to pass to environment
def clicks1(x):
    return clicks(x, params_clicks[0])
def clicks2(x):
    return clicks(x, params_clicks[1])
def clicks3(x):
    return clicks(x, params_clicks[2])

# Function wrapper to be able to pass to environment
def costs1(x):
    return costs(x, params_costs[0])
def costs2(x):
    return costs(x, params_costs[1])
def costs3(x):
    return costs(x, params_costs[2])

import warnings
warnings.simplefilter('ignore', UserWarning)

# experiments
gpts_rewards_per_experiment = [[],[],[]]
gpucb_rewards_per_experiment = [[],[],[]]

for e in tqdm(range(0,n_experiments)):  # cycle on experiments
  env = Environment_Multiple_Contexts(n_arms = n_prices,
                                        probabilities = p,
                                        bids = bids,
                                        average_number_of_clicks = [clicks1, clicks2, clicks3],
                                        average_cum_daily_cost = [costs1, costs2, costs3],
                                        noise_clicks = sigma_clicks,
                                        noise_cost = sigma_costs)
  # Create one pricing and advert. learner per context with the diff. prices and noises.
  pricing_learners_gpts = [TS_Pricing_Learner(n_arms = n_prices, prices = prices),
                           TS_Pricing_Learner(n_arms = n_prices, prices = prices),
                           TS_Pricing_Learner(n_arms = n_prices, prices = prices)]
  pricing_learners_gpucb =[TS_Pricing_Learner(n_arms = n_prices, prices = prices),
                           TS_Pricing_Learner(n_arms = n_prices, prices = prices),
                           TS_Pricing_Learner(n_arms = n_prices, prices = prices)]
  gpts_learners = [GPTS_Learner(n_arms = n_bids, arms = bids, alpha_clicks = sigma_clicks[0], alpha_costs = sigma_costs[0],
                              step = 10, normalize_y = False,plot = False, step_delay = 0), #n_restarts_optimizer = 9
                  GPTS_Learner(n_arms = n_bids, arms = bids, alpha_clicks = sigma_clicks[1], alpha_costs = sigma_costs[1],
                              step = 10, normalize_y = False,plot = False, step_delay = 0), #n_restarts_optimizer = 9
                  GPTS_Learner(n_arms = n_bids, arms = bids, alpha_clicks = sigma_clicks[2], alpha_costs = sigma_costs[2],
                              step = 10, normalize_y = False,plot = False, step_delay = 0) #n_restarts_optimizer = 9
                 ]
  gpucb_learners = [GPUCB_Learner(n_arms = n_bids, arms = bids, alpha_clicks = sigma_clicks[0], alpha_costs = sigma_costs[0],
                                step = 10, normalize_y = False, step_delay = 0), #plot = True, n_restarts_optimizer = 9,
                   GPUCB_Learner(n_arms = n_bids, arms = bids, alpha_clicks = sigma_clicks[1], alpha_costs = sigma_costs[1],
                                step = 10, normalize_y = False, step_delay = 0), #plot = True, n_restarts_optimizer = 9,
                   GPUCB_Learner(n_arms = n_bids, arms = bids, alpha_clicks = sigma_clicks[2], alpha_costs = sigma_costs[2],
                                step = 10, normalize_y = False, step_delay = 0) #plot = True, n_restarts_optimizer = 9,
                  ]

  gpts_daily_pricing_rewards = [np.array([]),np.array([]),np.array([])]
  gpucb_daily_pricing_rewards = [np.array([]),np.array([]),np.array([])]


  for t in range(0,T):  # cycle on time horizon

    # GPTS
    for i in range(3): # Each context (1/3, 1/3, 1/3)
        pulled_bid_ts = gpts_learners[i].pull_arm()  # pull bid for each context
        reward_click_ts, reward_cost_ts = env.bidding_round(pulled_bid_ts, i)
        gpts_learners[i].update(pulled_bid_ts, [reward_click_ts,reward_cost_ts])
        for k in range(reward_click_ts):
          pulled_price = pricing_learners_gpts[i].pull_arm()
          reward_price = env.pricing_round(pulled_price, i)
          pricing_learners_gpts[i].update(pulled_price, reward_price)
        gpts_daily_pricing_rewards[i] = np.append(gpts_daily_pricing_rewards[i], (reward_click_ts>0)*np.sum(pricing_learners_gpts[i].collected_rewards[-reward_click_ts:]))

    # GPUCB
    for i in range(3):
      pulled_bid_ucb = gpucb_learners[i].pull_arm()
      reward_click_ucb, reward_cost_ucb = env.bidding_round(pulled_bid_ucb, i)
      gpucb_learners[i].update(pulled_bid_ucb, [reward_click_ucb, reward_cost_ucb])
      for k in range(reward_click_ucb):
        pulled_price = pricing_learners_gpucb[i].pull_arm()
        reward_price = env.pricing_round(pulled_price, i)
        pricing_learners_gpucb[i].update(pulled_price, reward_price)
      gpucb_daily_pricing_rewards[i] = np.append(gpucb_daily_pricing_rewards[i],(reward_click_ucb>0)*np.sum(pricing_learners_gpucb[i].collected_rewards[-reward_click_ucb:]))
  for i in range(3):
    gpts_rewards_per_experiment[i].append(gpts_daily_pricing_rewards[i] - gpts_learners[i].collected_costs)
    gpucb_rewards_per_experiment[i].append(gpucb_daily_pricing_rewards[i] - gpucb_learners[i].collected_costs)

plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("t")
colors_context = ['b', 'r', 'g']
for i in range(3):
    opt = opt_rate[i]*prices[np.argmax(p[i]*prices)]*clicks(opt_bid[i], params_clicks[i]) - costs(opt_bid[i], params_costs[i])
    plt.plot(np.cumsum(np.mean(opt - gpts_rewards_per_experiment[i], axis = 0)), colors_context[i]+'.')
    plt.plot(np.cumsum(np.mean(opt - gpucb_rewards_per_experiment[i], axis = 0)), colors_context[i]+'-')
plt.legend(["GPTS_1","GPUCB_1","GPTS_2","GPUCB_2","GPTS_3","GPUCB_3"])
plt.show()

opt = np.sum([opt_rate[i]*prices[np.argmax(p[i]*prices)]*clicks(opt_bid[i], params_clicks[i]) - costs(opt_bid[i], params_costs[i]) for i in range(3)])


plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(opt - np.sum(gpts_rewards_per_experiment, axis=0), axis = 0)), 'r-')
plt.plot(np.cumsum(np.mean(opt - np.sum(gpucb_rewards_per_experiment, axis=0), axis = 0)), 'b-')
plt.legend(["GPTS","GPUCB"])
plt.show()

#np.savetxt('part4_1_gpts.txt',np.sum(gpts_rewards_per_experiment, axis=0))
#np.savetxt('part4_1_gpucb.txt',np.sum(gpucb_rewards_per_experiment, axis=0))

#gpts_rewards_per_experiment = np.loadtxt('part4_3_gpts.txt')
#gpucb_rewards_per_experiment = np.loadtxt('part4_3_gpucb.txt')

plt.figure()
plt.ylabel("Cumulative regret")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(opt - np.sum(gpts_rewards_per_experiment, axis=0), axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - np.sum(gpucb_rewards_per_experiment, axis=0), axis=0)), 'b')

time = range(0, T)
ts_std = np.std(np.cumsum(opt - np.sum(gpts_rewards_per_experiment, axis=0), axis=1), axis=0)
ts_metric= np.mean(np.cumsum(opt - np.sum(gpts_rewards_per_experiment, axis=0), axis=1), axis=0)
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([ts_metric - ts_std, (ts_metric + ts_std)[::-1]]),
         alpha=.5, fc='r', ec=None, label='standard deviation')

ucb_std = np.std(np.cumsum(opt - np.sum(gpucb_rewards_per_experiment, axis=0), axis=1), axis=0)
ucb_metric = np.mean(np.cumsum(opt - np.sum(gpucb_rewards_per_experiment, axis=0), axis=1), axis=0)
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([ucb_metric - ucb_std, (ucb_metric+ ucb_std)[::-1]]),
         alpha=.5, fc='b', ec=None, label='standard deviation')

plt.legend(["GPTS", "GPUCB"])
plt.show()

plt.figure()
plt.ylabel("Cumulative reward")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(np.sum(gpts_rewards_per_experiment, axis=0), axis=0)), 'r')
plt.plot(np.cumsum(np.mean(np.sum(gpucb_rewards_per_experiment, axis=0), axis=0)), 'b')

time = range(0, T)
ts_std = np.std(np.cumsum(np.sum(gpts_rewards_per_experiment, axis=0), axis=1), axis=0)
ts_metric= np.mean(np.cumsum(np.sum(gpts_rewards_per_experiment, axis=0), axis=1), axis=0)
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([ts_metric - ts_std, (ts_metric + ts_std)[::-1]]),
         alpha=.5, fc='r', ec=None, label='standard deviation')

ucb_std = np.std(np.cumsum(np.sum(gpucb_rewards_per_experiment, axis=0), axis=1), axis=0)
ucb_metric = np.mean(np.cumsum(np.sum(gpucb_rewards_per_experiment, axis=0), axis=1), axis=0)
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([ucb_metric - ucb_std, (ucb_metric+ ucb_std)[::-1]]),
         alpha=.5, fc='b', ec=None, label='standard deviation')

plt.legend(["GPTS", "GPUCB"])
plt.show()

plt.figure()
plt.ylabel("Instantaneous regret")
plt.xlabel("t")
plt.plot(np.mean(opt - np.sum(gpts_rewards_per_experiment, axis=0), axis=0), 'r')
plt.plot(np.mean(opt - np.sum(gpucb_rewards_per_experiment, axis=0), axis=0), 'b')

time = range(0, T)
ts_std = np.std(opt - np.sum(gpts_rewards_per_experiment, axis=0), axis=0)
ts_metric= np.mean(opt - np.sum(gpts_rewards_per_experiment, axis=0), axis=0)
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([ts_metric - ts_std, (ts_metric + ts_std)[::-1]]),
         alpha=.5, fc='r', ec=None, label='standard deviation')

ucb_std = np.std(opt - np.sum(gpucb_rewards_per_experiment, axis=0), axis=0)
ucb_metric = np.mean(opt - np.sum(gpucb_rewards_per_experiment, axis=0), axis=0)
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([ucb_metric - ucb_std, (ucb_metric+ ucb_std)[::-1]]),
         alpha=.5, fc='b', ec=None, label='standard deviation')

plt.legend(["GPTS", "GPUCB"])
plt.show()

plt.figure()
plt.ylabel("Instantaneous reward")
plt.xlabel("t")
plt.plot(np.mean(np.sum(gpts_rewards_per_experiment, axis=0), axis=0), 'r')
plt.plot(np.mean(np.sum(gpucb_rewards_per_experiment, axis=0), axis=0), 'b')

time = range(0, T)
ts_std = np.std(np.sum(gpts_rewards_per_experiment, axis=0), axis=0)
ts_metric= np.mean(np.sum(gpts_rewards_per_experiment, axis=0), axis=0)
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([ts_metric - ts_std, (ts_metric + ts_std)[::-1]]),
         alpha=.5, fc='r', ec=None, label='standard deviation')

ucb_std = np.std(np.sum(gpucb_rewards_per_experiment, axis=0), axis=0)
ucb_metric = np.mean(np.sum(gpucb_rewards_per_experiment, axis=0), axis=0)
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([ucb_metric - ucb_std, (ucb_metric+ ucb_std)[::-1]]),
         alpha=.5, fc='b', ec=None, label='standard deviation')

plt.legend(["GPTS", "GPUCB"])
plt.show()