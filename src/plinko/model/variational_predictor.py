from torch.distributions.normal import Normal
import torch.nn as nn
import torch
import torch.nn.functional as F
from .mlp import MLP
from ..misc import utils

class VariationalPredictor(nn.Module):
    def __init__(self,
                 env_size=11,
                 state_size=4,
                 env_embed_size=16,
                 state_embed_size=16,
                 encoder_hidden_sizes=[128, 128],
                 decoder_hidden_sizes=[128],
                 z_size=2):
        super().__init__()
        self.env_size = env_size
        self.state_size = state_size
        self.env_embed_size = env_embed_size
        self.state_embed_size = state_embed_size
        self.encoder_hidden_sizes = encoder_hidden_sizes
        self.decoder_hidden_sizes = decoder_hidden_sizes
        self.z_size = z_size

        self.env_embedder = MLP(input_size=env_size,
                                hidden_layer_size=None,
                                output_size=env_embed_size)
        self.state_embedder = MLP(input_size=state_size,
                                  hidden_layer_size=None,
                                  output_size=state_embed_size)
        self.encoder = MLP(input_size=[env_embed_size, state_embed_size],
                           hidden_layer_size=encoder_hidden_sizes,
                           output_size=[z_size, z_size],
                           activation=nn.ELU)
        self.decoder = MLP(input_size=z_size,
                           hidden_layer_size=decoder_hidden_sizes,
                           output_size=state_size,
                           activation=nn.ELU)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1, dtype=torch.float), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1, dtype=torch.float), requires_grad=False)

    def forward(self, envs, states):
        """
        envs: batch_size, env_size
        states: batch_size, t, state_size
        """
        batch_size, max_t, state_size = states.shape
        h_env = F.relu(self.env_embedder(envs))
        h_env = utils.expand_along_dim(h_env, max_t, 1)
        states = F.relu(self.state_embedder(states))

        mu, var = self.encoder(h_env, states)
        z = Normal(mu, var).rsample()
        return self.decoder(z)
