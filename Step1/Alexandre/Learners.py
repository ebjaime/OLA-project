import numpy as np


## Pricing


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


class TS_Pricing_Learner(Learner): # Thompson-Sampling (reward: number of purchases; actual_reward:  price*conversion_rate)
    def __init__(self,n_arms,prices):
        super().__init__(n_arms)                    # number of prices
        self.beta_parameters = np.ones((n_arms,2))  # parameters of beta distributions
        self.prices = prices                        # prices (array)
        self.actual_reward = np.array([])           # storage of price*conversion_rate

    def pull_arm(self):
        if(self.t < self.n_arms):
            return self.t, self.prices[self.t]
        sampled_conversion_rate = np.random.beta(self.beta_parameters[:,0],self.beta_parameters[:,1])
        idx = np.argmax(sampled_conversion_rate*self.prices)
        sampled_value_of_click = np.max(sampled_conversion_rate*self.prices)
        return idx, sampled_value_of_click

    # update parameters each time a reward (# of people that buy) is observed
    def update(self,pulled_arm, reward, n_clicks):
        self.t += 1
        self.update_observations(pulled_arm,reward)
        self.actual_reward = np.append(self.actual_reward, reward/n_clicks * self.prices[pulled_arm])
        self.beta_parameters[pulled_arm,0] = self.beta_parameters[pulled_arm,0] + reward
        self.beta_parameters[pulled_arm,1] = self.beta_parameters[pulled_arm,1] + n_clicks - reward


class UCB_Pricing_Learner(Learner): # UCB1 (reward: number of conversions; actual_reward:  price*conversion_rate)
    def __init__(self,n_arms,prices):
        super().__init__(n_arms)                              # number of arms/prices
        self.empirical_means = np.zeros(n_arms)               # mean reward for each arm (conversion rate)
        self.confidence = np.zeros(n_arms)                    # confidence bound for each arm
        self.prices = prices                                  # prices (array)
        self.n_clicks_per_arm = [[0] for i in range(n_arms)]  # number of total clicks for arm
        self.tot_n_clicks = 0                                 # cumulative number of clicks/rounds
        self.actual_reward = np.array([])                     # storage of price*conversion rate

    def pull_arm(self):
        if(self.t < self.n_arms):
            return self.t, self.prices[self.t]
        upper_bound = self.empirical_means + self.confidence
        pulled_arm = np.random.choice(np.where(upper_bound == upper_bound.max())[0])
        return pulled_arm, upper_bound.max()

    def update(self, pulled_arm, reward, n_clicks):
        self.t += 1
        self.tot_n_clicks += n_clicks
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm]* np.sum(self.n_clicks_per_arm[pulled_arm]) + reward*self.prices[pulled_arm] ) / (np.sum(self.n_clicks_per_arm[pulled_arm]) + n_clicks)
        self.confidence[pulled_arm] = self.prices[pulled_arm]*np.sqrt(2*np.log(self.tot_n_clicks)/(np.sum(self.n_clicks_per_arm[pulled_arm]) + n_clicks))
        self.n_clicks_per_arm[pulled_arm].append(n_clicks)
        self.actual_reward = np.append(self.actual_reward, reward/n_clicks * self.prices[pulled_arm])
        self.update_observations(pulled_arm, reward)


## Advertising (to fix)


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPTS_Learner(Learner):
  # constructor takes as input the number arms (bids), intializes to 0 their means and to an aye their standard deviation; we also need to store the arm that we pull at each round to fot the GPs
    def __init__(self, n_arms, arms):
        super().__init__(n_arms)
        self.arms = arms
        self.means_clicks = np.zeros(self.n_arms)
        self.means_costs = np.zeros(self.n_arms)
        self.sigmas_clicks = np.ones(self.n_arms) * 10
        self.sigmas_costs = np.ones(self.n_arms) * 10
        self.pulled_arms = []
        self.collected_clicks = []
        self.collected_costs = []
        # we initialize the parameters of the kernel and the two GPs
        alpha_clicks = 10  # standard deviation of the noise
        alpha_costs = 10   # standard deviation of the noise
        kernel_clicks = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3)) # kernel (squared exponential) with the range of the parameters
        kernel_costs = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))  # kernel (squared exponential) with the range of the parameters
        self.gp_clicks = GaussianProcessRegressor(kernel = kernel_clicks, alpha = alpha_clicks**2, n_restarts_optimizer = 10)  # (normalize_y = True)
        self.gp_costs = GaussianProcessRegressor(kernel = kernel_costs, alpha = alpha_costs**2, n_restarts_optimizer = 10)     # (normalize_y = True)

    # we also need to update the value of the least pulled arm (reward[0]: n_clicks, reward[1]: costs)
    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    # update the GP estimations and consequently the means and sigmas of each arm
    def update_model(self):
        # trining inputs and targets
        x = np.atleast_2d(self.pulled_arms).T
        ind_clicks = [(2*i) for i in range(int(len(self.collected_rewards)/2)) ]
        ind_costs = [(2*i+1) for i in range(int(len(self.collected_rewards)/2)) ]
        y_clicks = self.collected_rewards[ind_clicks]
        y_costs = self.collected_rewards[ind_costs]
        # fit the GP
        self.gp_clicks.fit(x,y_clicks)
        self.gp_costs.fit(x,y_costs)
        # update values of means and sigmas with the new predictions
        self.means_clicks, self.sigmas_clicks = self.gp_clicks.predict(np.atleast_2d(self.arms).T, return_std = True)
        self.means_costs, self.sigmas_costs = self.gp_costs.predict(np.atleast_2d(self.arms).T, return_std = True)
        self.sigmas_clicks = np.maximum(self.sigmas_clicks, 1e-2)  # force sigmas > 0
        self.sigmas_costs = np.maximum(self.sigmas_costs, 1e-2)  # force sigmas > 0

    # functions that calls the functions implemented above (reward[0]: n_clicks, reward[1]: costs)
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.collected_clicks.append(reward[0])
        self.collected_costs.append(reward[1])
        self.update_model()

    # function in which the learner chooses the arm to pull at each round
    def pull_arm(self):
        # returns index of the maximum value drawn from the arm normal distribution
        sampled_clicks = np.random.normal(self.means_clicks, self.sigmas_clicks)
        sampled_costs = np.random.normal(self.means_costs, self.sigmas_costs)
        return np.argmax(sampled_clicks - sampled_costs)