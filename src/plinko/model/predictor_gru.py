import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import MLP
from ..misc import utils
from ..misc.gaussianmixture import GaussianMixture
import sys

epsilon = sys.float_info.epsilon


class GRUPredictor(nn.Module):

    def __init__(self,
                 env_size,
                 state_size,
                 env_embed_size=32,
                 state_embed_size=16,
                 num_gaussians=8,
                 num_rnn = 2,
                 trainable_h0=False
                 ):
        super(GRUPredictor, self).__init__()
        self.env_size = env_size
        self.state_size = state_size
        self.env_embed_size = env_embed_size
        self.num_gaussians = num_gaussians
        self.num_rnn = num_rnn
        self.hidden_size = 128
        self.trainable_h0 = trainable_h0

        self.env_embedder = MLP(input_size=env_size,
                                hidden_layer_size=None,
                                output_size=env_embed_size)
        self.state_embedder = MLP(input_size=state_size,
                                  hidden_layer_size=None,
                                  output_size=state_embed_size)
        self.gru = nn.GRU(input_size=env_embed_size + state_embed_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_rnn)
        self.mlp = MLP(input_size=self.hidden_size,
                       hidden_layer_size=self.hidden_size,
                       output_size=[self.num_gaussians,  # alpha
                                    self.num_gaussians * 2,  # mu
                                    self.num_gaussians * 3])  # sigma

        if self.trainable_h0:
            self.register_parameter('init_gru_h',
                                    torch.nn.Parameter(torch.rand(self.gru.num_layers, self.hidden_size)))

    def predict_using_true_states(self, h_env, states):
        """
        At each t, given the true state at t, predict the distribution for t+1
        """
        batch_size, t, state_size = states.shape

        if self.trainable_h0:
            h_n = utils.expand_along_dim(self.init_gru_h, batch_size, 1).contiguous()
        else:
            h_n = torch.zeros(self.gru.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=h_env.device)

        h_env = utils.expand_along_dim(h_env, t, 0)
        states = states.permute(1, 0, 2)
        states = self.state_embedder(states)
        h = torch.cat([h_env, states], dim=-1)

        h, h_n = self.gru(h, h_n)
        a, m, s = self.mlp(h)
        a = a.permute(1, 0, 2)
        m = m.permute(1, 0, 2)
        s = s.permute(1, 0, 2)
        a = F.softmax(a, dim=-1)
        s = F.softplus(s) + epsilon
        m = m.view(a.shape + (2,))
        s = s.view(a.shape + (3,))
        gm = GaussianMixture(a, m, s, lower_cholesky=True)
        return gm, h_n, a[:, -1], m[:, -1], s[:, -1]

    def predict_using_sampled_states(self, h_env, h_n, a, m, s, predict_t):
        """
        At each t, samples a state at t to predict the distribution for t+1
        """
        gm = GaussianMixture(a, m, s, lower_cholesky=True)
        state = gm.sample()

        gms = [gm]
        samples = [state]
        for i in range(predict_t - 1):
            state = self.state_embedder(state)
            h = torch.cat([h_env, state], dim=-1).unsqueeze(0)
            h, h_n = self.gru(h, h_n)
            a, m, s = self.mlp(h.squeeze(0))
            a = F.softmax(a, dim=-1)
            s = F.softplus(s) + epsilon
            m = m.view(a.shape + (2,))
            s = s.view(a.shape + (3,))
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
        h_env = self.env_embedder(envs)
        h_env = F.relu(h_env)
        inter_gm, h_n, a, m, s = self.predict_using_true_states(h_env, states)

        # predict max_t future states by sampling from last GM
        if predict_t > 0:
            extra_gm, samples = self.predict_using_sampled_states(h_env, h_n, a, m, s, predict_t)
            return inter_gm, extra_gm, samples
        else:
            return inter_gm


class GRUPredictor_mu(nn.Module):
    """
    GRU model to predict the mu only
    """
    def __init__(self,
                 env_size,
                 state_size,
                 env_embed_size=32,
                 state_embed_size=16,
                 num_gaussians=8,
                 trainable_h0=False
                 ):
        super(GRUPredictor_mu, self).__init__()
        self.env_size = env_size
        self.state_size = state_size
        self.env_embed_size = env_embed_size
        self.num_gaussians = num_gaussians
        self.hidden_size = 128
        self.trainable_h0 = trainable_h0

        self.env_embedder = MLP(input_size=env_size,
                                hidden_layer_size=None,
                                output_size=env_embed_size)
        self.state_embedder = MLP(input_size=state_size,
                                  hidden_layer_size=None,
                                  output_size=state_embed_size)
        self.gru = nn.GRU(input_size=env_embed_size + state_embed_size,
                          hidden_size=self.hidden_size,
                          num_layers=2)
        self.mlp = MLP(input_size=self.hidden_size,
                       hidden_layer_size=self.hidden_size,
                       output_size=[self.num_gaussians,  # alpha
                                    self.num_gaussians * 2,  # mu
                                    self.num_gaussians * 3])  # sigma

        if self.trainable_h0:
            self.register_parameter('init_gru_h',
                                    torch.nn.Parameter(torch.rand(self.gru.num_layers, self.hidden_size)))

    def predict_using_true_states(self, h_env, states):
        """
        At each t, given the true state at t, predict the mu for t+1
        """
        batch_size, t, state_size = states.shape

        if self.trainable_h0:
            h_n = utils.expand_along_dim(self.init_gru_h, batch_size, 1).contiguous()
        else:
            h_n = torch.zeros(self.gru.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=h_env.device)

        h_env = utils.expand_along_dim(h_env, t, 0)
        states = states.permute(1, 0, 2)
        states = self.state_embedder(states)
        h = torch.cat([h_env, states], dim=-1)

        h, h_n = self.gru(h, h_n)
        a, m, s = self.mlp(h)
        a = a.permute(1, 0, 2)
        m = m.permute(1, 0, 2)
        s = s.permute(1, 0, 2)
        a = F.softmax(a, dim=-1)
        s = F.softplus(s) + epsilon
        m = m.view(a.shape + (2,))
        s = s.view(a.shape + (3,))
        gm = GaussianMixture(a, m, s, lower_cholesky=True)
        return gm, h_n, a[:, -1], m[:, -1], s[:, -1] # why return the last element?

    def predict_using_sampled_states(self, h_env, h_n, a, m, s, predict_t):
        """
        At each t, samples a state at t to predict the mu for t+1
        """
        gm = GaussianMixture(a, m, s, lower_cholesky=True)
        # state = gm.sample()
#         print("sample is ", gm.sample())
#         print(gm.sample().shape)
        # returns the mu from the Gaussian mixture
#         print("mu is ", gm.mu[:, 0])
#         print(gm.mu[:, 0].shape)
        state = gm.mu[:, 0]

        gms = [gm]
        samples = [state]
        for i in range(predict_t - 1):
            state = self.state_embedder(state)
            # print('h_env shape:', h_env.shape)
            # print('state shape:', state.shape)
            h = torch.cat([h_env, state], dim=-1).unsqueeze(0)
            h, h_n = self.gru(h, h_n)
            a, m, s = self.mlp(h.squeeze(0))
            a = F.softmax(a, dim=-1)
            s = F.softplus(s) + epsilon
            m = m.view(a.shape + (2,))
            s = s.view(a.shape + (3,))
            gm = GaussianMixture(a, m, s, lower_cholesky=True)
            # state = gm.sample()
            # returns the mu from the Gaussian mixture
            state = gm.mu[:, 0]
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
        h_env = self.env_embedder(envs)
        h_env = F.relu(h_env)
        inter_gm, h_n, a, m, s = self.predict_using_true_states(h_env, states)

        # predict max_t future states by sampling from last GM
        if predict_t > 0:
            extra_gm, samples = self.predict_using_sampled_states(h_env, h_n, a, m, s, predict_t)
            return inter_gm, extra_gm, samples
        else:
            return inter_gm


class GRUPredictor_determ(nn.Module):
    """
    Deterministic GRU model to predict position and velocity
    """
    def __init__(self,
                 env_size,
                 state_size,
                 env_embed_size=32,
                 state_embed_size=32,
                 gru_hidden_size=64,
                 num_gaussians=8,
                 trainable_h0=False
                 ):
        super(GRUPredictor_determ, self).__init__()
        self.env_size = env_size
        self.state_size = state_size
        self.env_embed_size = env_embed_size
        self.num_gaussians = num_gaussians
        self.hidden_size = gru_hidden_size
        self.trainable_h0 = trainable_h0

        self.env_embedder = MLP(input_size=env_size,
                                hidden_layer_size=None,
                                output_size=env_embed_size)
        self.state_embedder = MLP(input_size=state_size,
                                  hidden_layer_size=None,
                                  output_size=state_embed_size)
        self.gru = nn.GRU(input_size=env_embed_size + state_embed_size,
                          hidden_size=self.hidden_size,
                          num_layers=2)
        self.mlp = MLP(input_size=self.hidden_size,
                       hidden_layer_size=self.hidden_size,
                       output_size=[2,  # px, py
                                    2]) # vx, vy

        self.col_classifier = MLP(input_size=[env_size, state_size - 2],
            hidden_layer_size=[32, 32, 32, 32, 32],
            activation=F.elu,
            output_size=7)

        if self.trainable_h0:
            self.register_parameter('init_gru_h',
                                    torch.nn.Parameter(torch.rand(self.gru.num_layers, self.hidden_size)))

    def predict_using_true_states(self, h_env, states, env):
        """
        At each t, given the true state at t, predict the mu for t+1
        """
        # print('env shape', env.shape)
        # print('states shape', states.shape)
        env = env.unsqueeze(1)
        # print('env shape', env.shape)
        envs = env.expand(env.shape[0], states.shape[1], env.shape[-1])
        # print('envs shape', envs.shape)
        # print('envs ', envs[0])
        col_outputs = self.col_classifier(envs.reshape(-1, envs.shape[-1]), states.reshape(-1, states.shape[-1])[:, :4])
        # col_outputs = col_outputs.view(states.shape[0], states.shape[1], -1)
        col_pred = col_outputs.argmax(-1)
        col_pred = col_pred.view(states.shape[0], states.shape[1], -1)
        col_pred = torch.tensor(col_pred, dtype=torch.float, device=h_env.device)
        # print("col_pred shape", col_pred.shape)
        
        batch_size, t, state_size = states.shape

        if self.trainable_h0:
            h_n = utils.expand_along_dim(self.init_gru_h, batch_size, 1).contiguous()
        else:
            h_n = torch.zeros(self.gru.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=h_env.device)

        h_env = utils.expand_along_dim(h_env, t, 0)
        next_t = states[:, :, state_size - 1] + 1
        next_t = next_t.view(next_t.shape[0], next_t.shape[1], -1)
#         new_states = torch.cat([states[:, :, :4], col_pred, states[:, :, 5].view(states.shape[0], states.shape[1], -1)], dim=-1).clone()
        new_states = states.permute(1, 0, 2)

        new_states = self.state_embedder(new_states)
        h = torch.cat([h_env, new_states], dim=-1)

        h, h_n = self.gru(h, h_n)
        p, v = self.mlp(h)
        p = p.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        # print('p shape is ', p.shape)
        # print('v shape is ', v.shape)
        return h_n, p, v, next_t, col_outputs, col_pred

    def predict_using_sampled_states(self, h_env, h_n, p, v, t, env, predict_t):
        """
        At each t, samples a state at t to predict the mu for t+1
        """
        state = torch.cat([p[:, -1], v[:, -1]], dim = -1)
        col_outputs = self.col_classifier(env, state[:, :4])
        col_pred = col_outputs.argmax(-1)
        col_pred = col_pred.view(state.shape[0], -1)
        col_pred = torch.tensor(col_pred, dtype=torch.float, device=h_env.device)
        state = torch.cat([p[:, -1], v[:, -1], col_pred, t[:, -1]], dim = -1)

        samples_p = [p[:, -1]]
        samples_v = [v[:, -1]]
        for i in range(predict_t - 1):
            # env = env.unsqueeze(1)
            # envs = env.expand(env.shape[0], state.shape[1], env.shape[-1])
            # print('envs.shape', env.shape)
            # print('state.shape', state.shape)
            # next_col = next_col.view(next_col.shape[0], -1)
            next_t = t[:, -1] + 1
            next_t = next_t.view(next_t.shape[0], -1)
            state = self.state_embedder(state)
#             print('h_env shape:', h_env.shape)
#             print('state shape:', state.shape)
            h = torch.cat([h_env, state], dim=-1).unsqueeze(0)
            h, h_n = self.gru(h, h_n) # p, v at next timepoint
            p, v = self.mlp(h.squeeze(0))
            state = torch.cat([p, v], dim = -1)
            col_outputs = self.col_classifier(env, state[:, :4]) # collision at next timepoint
            col_pred = col_outputs.argmax(-1)
            col_pred = col_pred.view(state.shape[0], -1)
            col_pred = torch.tensor(col_pred, dtype=torch.float, device=h_env.device)
            state = torch.cat([p, v, col_pred, next_t], dim=-1) # state at next timepoint
            # print('p shape is ', p.shape)
            # print('v shape is ', v.shape)            
            # print('t shape is ', next_t.shape)
            samples_p.append(p)
            samples_v.append(v)
        return torch.stack(samples_p, dim=1), torch.stack(samples_v, dim=1)

    def forward(self, envs, states, predict_t=0):
        """
        envs: batch_size, env_size
        states: batch_size, t, state_size
        if predict_t is 0, only returns predicted GM using true states at each t
        if predict_t > 0, returns GM using true states, GM using sampled states, and samples
        """
        h_env = self.env_embedder(envs)
        h_env = F.relu(h_env)
        h_n, p, v, next_t, col_output, col_pred = self.predict_using_true_states(h_env, states, envs)

        # predict max_t future states by sampling from last GM
        if predict_t > 0:
            samples_p, samples_v = self.predict_using_sampled_states(h_env, h_n, p, v, next_t, envs, predict_t)
            return samples_p, samples_v
        else:
            return p, v, next_t, col_output
        

class GRUPredictor_determ_p(nn.Module):
    """
    Deterministic GRU model to predict position and velocity
    """
    def __init__(self,
                 env_size,
                 state_size,
                 env_embed_size=32,
                 state_embed_size=32,
                 gru_hidden_size=64,
                 num_gaussians=8,
                 trainable_h0=False
                 ):
        super(GRUPredictor_determ_p, self).__init__()
        self.env_size = env_size
        self.state_size = state_size
        self.env_embed_size = env_embed_size
        self.num_gaussians = num_gaussians
        self.hidden_size = gru_hidden_size
        self.trainable_h0 = trainable_h0

        self.env_embedder = MLP(input_size=env_size,
                                hidden_layer_size=None,
                                output_size=env_embed_size)
        self.state_embedder = MLP(input_size=state_size,
                                  hidden_layer_size=None,
                                  output_size=state_embed_size)
        self.gru = nn.GRU(input_size=env_embed_size + state_embed_size,
                          hidden_size=self.hidden_size,
                          num_layers=2)
        self.mlp = MLP(input_size=self.hidden_size,
                       hidden_layer_size=self.hidden_size,
                       output_size=2)

        self.col_classifier = MLP(input_size=[env_size, state_size - 2],
            hidden_layer_size=[32, 32, 32, 32, 32],
            activation=F.elu,
            output_size=7)

        if self.trainable_h0:
            self.register_parameter('init_gru_h',
                                    torch.nn.Parameter(torch.rand(self.gru.num_layers, self.hidden_size)))

    def predict_using_true_states(self, h_env, states, env):
        """
        At each t, given the true state at t, predict the mu for t+1
        """
        # print('env shape', env.shape)
        # print('states shape', states.shape)
        env = env.unsqueeze(1)
        # print('env shape', env.shape)
        envs = env.expand(env.shape[0], states.shape[1], env.shape[-1])
        # print('envs shape', envs.shape)
        # print('envs ', envs[0])
        col_outputs = self.col_classifier(envs.reshape(-1, envs.shape[-1]), states.reshape(-1, states.shape[-1])[:, :2])
        # col_outputs = col_outputs.view(states.shape[0], states.shape[1], -1)
        col_pred = col_outputs.argmax(-1)
        col_pred = col_pred.view(states.shape[0], states.shape[1], -1)
        col_pred = torch.tensor(col_pred, dtype=torch.float, device=h_env.device)
        # print("col_pred shape", col_pred.shape)
        
        batch_size, t, state_size = states.shape

        if self.trainable_h0:
            h_n = utils.expand_along_dim(self.init_gru_h, batch_size, 1).contiguous()
        else:
            h_n = torch.zeros(self.gru.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=h_env.device)

        h_env = utils.expand_along_dim(h_env, t, 0)
        next_t = states[:, :, state_size - 1] + 1
        next_t = next_t.view(next_t.shape[0], next_t.shape[1], -1)
        new_states = states.permute(1, 0, 2)

        new_states = self.state_embedder(new_states)
        h = torch.cat([h_env, new_states], dim=-1)

        h, h_n = self.gru(h, h_n)
        p = self.mlp(h)
        p = p.permute(1, 0, 2)
        # print('p shape is ', p.shape)
        # print('v shape is ', v.shape)
        return h_n, p, next_t, col_outputs, col_pred

    def predict_using_sampled_states(self, h_env, h_n, p, t, env, predict_t):
        """
        At each t, samples a state at t to predict the mu for t+1
        """
        state = torch.cat([p[:, -1]], dim = -1)
        col_outputs = self.col_classifier(env, state[:, :2])
        col_pred = col_outputs.argmax(-1)
        col_pred = col_pred.view(state.shape[0], -1)
        col_pred = torch.tensor(col_pred, dtype=torch.float, device=h_env.device)
        state = torch.cat([p[:, -1], col_pred, t[:, -1]], dim = -1)

        samples_p = [p[:, -1]]
        for i in range(predict_t - 1):
            # env = env.unsqueeze(1)
            # envs = env.expand(env.shape[0], state.shape[1], env.shape[-1])
            # print('envs.shape', env.shape)
            # print('state.shape', state.shape)
            # next_col = next_col.view(next_col.shape[0], -1)
            next_t = t[:, -1] + 1
            next_t = next_t.view(next_t.shape[0], -1)
            state = self.state_embedder(state)
#             print('h_env shape:', h_env.shape)
#             print('state shape:', state.shape)
            h = torch.cat([h_env, state], dim=-1).unsqueeze(0)
            h, h_n = self.gru(h, h_n) # p, v at next timepoint
            p = self.mlp(h.squeeze(0))
            state = torch.cat([p], dim = -1)
            col_outputs = self.col_classifier(env, state[:, :2]) # collision at next timepoint
            col_pred = col_outputs.argmax(-1)
            col_pred = col_pred.view(state.shape[0], -1)
            col_pred = torch.tensor(col_pred, dtype=torch.float, device=h_env.device)
            state = torch.cat([p, col_pred, next_t], dim=-1) # state at next timepoint
            # print('p shape is ', p.shape)
            # print('v shape is ', v.shape)            
            # print('t shape is ', next_t.shape)
            samples_p.append(p)
        return torch.stack(samples_p, dim=1)

    def forward(self, envs, states, predict_t=0):
        """
        envs: batch_size, env_size
        states: batch_size, t, state_size
        if predict_t is 0, only returns predicted GM using true states at each t
        if predict_t > 0, returns GM using true states, GM using sampled states, and samples
        """
        h_env = self.env_embedder(envs)
        h_env = F.relu(h_env)
        h_n, p, next_t, col_output, col_pred = self.predict_using_true_states(h_env, states, envs)

        # predict max_t future states by sampling from last GM
        if predict_t > 0:
            samples_p = self.predict_using_sampled_states(h_env, h_n, p, next_t, envs, predict_t)
            return samples_p
        else:
            return p, next_t, col_output