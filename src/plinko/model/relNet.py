import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import MLP
from ..misc import utils
from ..misc.gaussianmixture import GaussianMixture
import sys
import numpy as np

epsilon = epsilon = sys.float_info.epsilon

# consider changing last layer to permutation layer

class relNet(nn.Module):
    # create embedder and objects
    def __init__(self,
                 state_size = 4, state_hn = 4, state_embed_size=32,
                 obj_size = 3, obj_n = 3, obj_hn = 4, obj_embed_size=16,
                 nullobj_n = 3,
                 mlp_hsize = 48, mlp_hn=3, mlp_outsize=1,
                 ):
        super(relNet, self).__init__()

        # state embedder
        self.state_size = state_size
        self.state_embed_size = state_embed_size
        self.state_hn = state_hn

        self.state_embdder = MLP(input_size=state_size,
                                    hidden_layer_size=state_hn*[state_embed_size],
                                    output_size=state_embed_size)

        # different embedding for each object
        self.obj_size = obj_size
        self.obj_n = obj_n
        self.obj_embed_size = obj_embed_size
        self.obj_hn = obj_hn

        self.obj_embdder = nn.ModuleList()
        for i in range(obj_n):
            self.obj_embdder.append(nn.ModuleList())
            self.obj_embdder[i].append(MLP(input_size=obj_size,
                                        hidden_layer_size=obj_hn*[obj_embed_size],
                                        output_size=obj_embed_size,
                                        activation = F.elu))
            #self.obj_embdder[i].append(MLP(input_size=obj_size,
            #                               hidden_layer_size=obj_hn * [obj_embed_size],
            #                               output_size=obj_embed_size,
            #                               activation=F.elu))

        # null embeddings (i.e. 2 walls, ground), constant across environments
        self.nullobj_n = nullobj_n

        self.register_parameter('null_embed',
                                torch.nn.Parameter(torch.rand(nullobj_n,obj_embed_size)))

        # relational coding (same for all objects)
        self.mlp_hsize = mlp_hsize
        self.mlp_hn = mlp_hn
        self.mlp_outsize = mlp_outsize

        self.mlp = MLP(input_size = [state_embed_size, obj_embed_size],
                        hidden_layer_size=mlp_hn*[mlp_hsize],
                        output_size=mlp_outsize,
                        activation=F.elu)

        # "permutation" layer
        self.scores = PermuteLayer(nullobj_n+obj_n)

    def forward(self, envs, states, predict_t=0):
        """
        envs: batch_size, obj_size*obj_n (in order)
        states: batch_size, state_size
        """

        Nbatch = list(envs.shape[:-1])[0]
        scores = []
        # state embedder
        states_embd = self.state_embdder(states)

        # for each object
        for i_obj in range(self.obj_n):
            sidx = self.obj_size*i_obj
            x = envs[:,sidx:(sidx+self.obj_size)]

            # get embedding
            for i_layer in range(len(self.obj_embdder[i_obj])):
                x = self.obj_embdder[i_obj][i_layer](x)

            # feed it through mlp
            x = self.mlp(states_embd,x)
            scores.append(x)

        # for null objects
        for i_nullobj in range(self.nullobj_n):
            x = self.null_embed[i_nullobj,:].view(1,-1).repeat(Nbatch,1)
            x = self.mlp(states_embd,x)
            scores.append(x)

        out = torch.cat(scores, -1)
        out = self.scores(out)

        return out


class PermuteLayer(nn.Module):
    # from cs236hw3
    """Layer to permute the ordering of inputs.

    Because our data is 2-D, forward() and inverse() will reorder the data in the same way.
    """

    def __init__(self, num_inputs):
        super(PermuteLayer, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])

    def forward(self, inputs):
        return inputs[..., self.perm]

    def inverse(self, inputs):
        return inputs[..., self.perm]
