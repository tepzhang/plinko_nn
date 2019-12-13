import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import MLP
from ..misc import utils
from ..misc.gaussianmixture import GaussianMixture
import sys

epsilon = epsilon = sys.float_info.epsilon


class baseMLP(nn.Module):

    def __init__(self,
                 state_size=4, env_size=9,
                 h_size = 144, h_n = 10,
                 num_gaussians=4
                 ):
        super(baseMLP, self).__init__()

        self.env_size = env_size
        self.state_size = state_size
        self.h_size = h_size
        self.h_n = h_n
        self.num_gaussians = num_gaussians
        self.out_msize = state_size
        self.out_ssize = int((state_size*state_size+state_size)/2)
        self.output_size = [self.num_gaussians,  # alpha
                            self.num_gaussians * self.out_msize,  # mu
                            self.num_gaussians * self.out_ssize]

        self.mlp = MLP(input_size=self.env_size+self.state_size,
                       hidden_layer_size=h_n*[h_size],
                       output_size=self.output_size)  # sigma

    def predict_using_true_states(self, h_env, states):
        """
        At each t, given the true state at t, predict the distribution for t+1
        """
        batch_size, t, state_size = states.shape

        h_env = utils.expand_along_dim(h_env, t, 0)
        states = states.permute(1, 0, 2)
        h = torch.cat([h_env, states], dim=-1)
        a, m, s = self.mlp(h)

        a = a.permute(1, 0, 2)
        m = m.permute(1, 0, 2)
        s = s.permute(1, 0, 2)
        a = F.softmax(a, dim=-1)
        s = F.softplus(s) + epsilon
        m = m.view(a.shape + (self.out_msize,))
        s = s.view(a.shape + (self.out_ssize,))
        gm = GaussianMixture(a, m, s, lower_cholesky=True)
        return gm, a[:, -1], m[:, -1], s[:, -1]

    def predict_using_sampled_states(self, env, a, m, s, predict_t):
        """
        At each t, samples a state at t to predict the distribution for t+1
        """
        gm = GaussianMixture(a, m, s, lower_cholesky=True)
        state = gm.sample()

        gms = [gm]
        samples = [state]
        for i in range(predict_t - 1):
            h = torch.cat([env, state], dim=-1).unsqueeze(0)
            a, m, s = self.mlp(h.squeeze(0))

            a = F.softmax(a, dim=-1)
            s = F.softplus(s) + epsilon
            m = m.view(a.shape + (self.out_msize,))
            s = s.view(a.shape + (self.out_ssize,))
            gm = GaussianMixture(a, m, s, lower_cholesky=True)
            state = gm.sample()
            samples.append(state)
            gms.append(gm)
        return gms, torch.stack(samples, dim=1)

    def forward(self, envs, states, predict_t=0):
        """
        envs: batch_size, env_size
        states: batch_size, t, state_size
        if predict_t is 0, only returns predicted GM using true states at each t
        if predict_t > 0, returns GM using true states, GM using sampled states, and samples
        """
        inter_gm, a, m, s = self.predict_using_true_states(envs, states)

        # predict max_t future states by sampling from last GM
        if predict_t > 0:
            extra_gm, samples = self.predict_using_sampled_states(envs, a, m, s, predict_t)
            return inter_gm, extra_gm, samples
        else:
            return inter_gm
