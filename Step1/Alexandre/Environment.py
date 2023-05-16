## Environment

import numpy as np

class Environment():

    def __init__(self, n_arms, probabilities, bids, average_number_of_clicks, average_cum_daily_cost, noise_clicks, noise_cost):
        self.n_arms = n_arms                                            # number of prices
        self.probabilities = probabilities                              # conversion rates for every price/arm
        self.bids = bids                                                # bids
        self.average_number_of_clicks = average_number_of_clicks        # curve of average number of clicks (y = f(bids))
        self.average_cum_daily_cost = average_cum_daily_cost            # curve of cumulative daily cost (y = g(bids))
        self.noise_clicks = noise_clicks                                # gaussian noise for the average number of clicks sampling
        self.noise_cost = noise_cost                                    # gaussian noise for the cumulative daily cost sampling

  # daily rewards
    def round(self, pulled_price, pulled_bid):
        reward_click = int(np.random.normal(self.average_number_of_clicks(self.bids[pulled_bid]),self.noise_clicks))        # number of people that click on the ad
        reward_price = np.random.binomial(reward_click,self.probabilities[pulled_price])                               # number of people that buy once they clicked
        reward_cost = np.random.normal(self.average_cum_daily_cost(self.bids[pulled_bid]),self.noise_cost)                  # cumulative daily cost

        return reward_price, reward_click, reward_cost