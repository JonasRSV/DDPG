import numpy as np

class Noise(object):
    #####################################################################
    # PR for something like this would be nice                          #
    # https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process  #
    #####################################################################

    def __init__(self, sigma, mu, reversion):
        self.sigma     = sigma
        self.mu        = mu
        self.reversion = reversion


    def process(self, noise):
        self.reversion = self.reversion * self.reversion
        return np.random.normal(loc=self.mu, scale=self.sigma) * self.reversion



