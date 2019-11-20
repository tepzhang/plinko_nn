import torch
from torch.distributions.multivariate_normal import MultivariateNormal, _batch_mahalanobis
from torch.distributions.categorical import Categorical
from ..misc import utils
import math
import sys


epsilon = sys.float_info.epsilon


def to_lower_triangle(t, d):
    m = torch.zeros(t.shape[:-1] + (d, d), device=t.device)
    last = 0
    for i in range(d):
        m[...,i,:i+1] = t[...,last:last+i+1]
        last += i + 1
    return m


class GaussianMixture:

    def __init__(self, alpha, mu, sigma, lower_cholesky=False):
        """
        alpha: [..., n]
        mu: [..., n, d]
        sigma: [..., n, d, d] # unless lower_vector, then [..., n, (d^2 - d)/2 + d]
        If lower_vector is true, sigma is treated as a vector encoding the lower triangle
            where indices map to 0 -> (0, 0), 1 -> (1, 0), 2 -> (1, 1), 3 -> (2, 0), ...
        """
        d = mu.shape[-1]
        assert sigma.shape[-2:] == (d, d) or (lower_cholesky and sigma.shape[-1] == (d ** 2 - d) / 2 + d)
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma
        self.lower_cholesky = lower_cholesky
        self.mvn_sample = None  # pytorch allows for batch sampling
        self.mvn_logp = None  # pytorch does not allow for batch logp

        if sigma.shape[-1] == (d ** 2 - d) / 2 + d:  # lower cholesky vector
            self.sigma = to_lower_triangle(self.sigma, d)

        if self.lower_cholesky:
            self.mvn = MultivariateNormal(self.mu, scale_tril=self.sigma)
        else:
            self.mvn = MultivariateNormal(self.mu, self.sigma)

        self.categorical = Categorical(self.alpha)

    @property
    def num_gaussians(self):
        return self.alpha.shape[-1]

    @property
    def point_dim(self):
        return self.mu.shape[-1]

    def log_p(self, x):
        """
        x: tensor with shape [batch_size, d]
        return: vector with length batch_size
        """

        x = utils.expand_along_dim(x, self.num_gaussians, -1)
        diff = x - self.mu
        M = _batch_mahalanobis(self.mvn._unbroadcasted_scale_tril, diff)
        half_log_det = self.mvn._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        logp = -0.5 * (self.point_dim * math.log(2 * math.pi) + M) - half_log_det
        probs = logp.exp()
        weighted_probs = self.alpha * probs
        mixed_probs = weighted_probs.sum(-1)
        return (mixed_probs + epsilon).log()

    def sample(self):
        """
        Returns tensor with shape [..., d]
        """
        dist_i = self.categorical.sample()
        s = self.mvn.sample()
        return utils.select_along_dim(s, dist_i)

    def __repr__(self):
        return 'GaussianMixture with sample shape=' + str(tuple(self.mu.shape[:-2]) + (self.mu.shape[-1],))
