import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class Learner:
    def __init__(self,n_arms):
        self.n_arms = n_arms
        self.t = 0                                              # current round value
        self.rewards_per_arm = [[] for i in range(n_arms)]  # value of collected rewards for each round and for each arm
        self.collected_rewards = np.array([])                   # values of collected rewards for each round

    # function that updates the observation's list once the reward is returned by the environment
    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)


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
        self.collected_buyers = np.array([])
        self.collected_clicks = np.array([])
        self.collected_costs = np.array([])
        # we initialize the parameters of the kernel and the two GPs
        alpha_clicks = 10  # standard deviation of the noise
        alpha_costs = 10   # standard deviation of the noise
        kernel_clicks = C(1.0, (1e-3, 1e8)) * RBF(1.0, (1e-3, 1e8)) # kernel (squared exponential) with the range of the parameters
        kernel_costs = C(1.0, (1e-3, 1e8)) * RBF(1.0, (1e-3, 1e8))  # kernel (squared exponential) with the range of the parameters
        self.gp_clicks = GaussianProcessRegressor(kernel=kernel_clicks, alpha=alpha_clicks**2,
                                                  n_restarts_optimizer=5)  # (normalize_y = True)
        self.gp_costs = GaussianProcessRegressor(kernel=kernel_costs, alpha=alpha_costs**2,
                                                 n_restarts_optimizer=5)     # (normalize_y = True)

    # we also need to update the value of the least pulled arm (reward[0]: n_clicks, reward[1]: costs)
    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    # update the GP estimations and consequently the means and sigmas of each arm
    def update_model(self):
        # trining inputs and targets
        x = np.atleast_2d(self.pulled_arms).T
        y_clicks = self.collected_clicks
        y_costs = self.collected_costs
        # fit the GP
        self.gp_clicks.fit(x,y_clicks)
        self.gp_costs.fit(x,y_costs)
        # update values of means and sigmas with the new predictions
        self.means_clicks, self.sigmas_clicks = self.gp_clicks.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.means_costs, self.sigmas_costs = self.gp_costs.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas_clicks = np.maximum(self.sigmas_clicks, 1e-2)  # force sigmas > 0
        self.sigmas_costs = np.maximum(self.sigmas_costs, 1e-2)  # force sigmas > 0

    # functions that calls the functions implemented above (reward[0]: n_clicks, reward[1]: costs, reward[2]: number of buyers)
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward[0:1])
        self.collected_clicks = np.append(self.collected_clicks, reward[0])
        self.collected_costs = np.append(self.collected_costs, reward[1])
        self.collected_buyers = np.append(self.collected_buyers, reward[2])
        self.update_model()

    # function in which the learner chooses the arm to pull at each round
    def pull_arm(self, opt_value_of_click):
        # returns index of the maximum value drawn from the arm normal distribution
        sampled_clicks = np.random.normal(self.means_clicks, self.sigmas_clicks)
        sampled_costs = np.random.normal(self.means_costs, self.sigmas_costs)
        return np.argmax(opt_value_of_click*sampled_clicks - sampled_costs)