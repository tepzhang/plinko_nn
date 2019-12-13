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

class relRnn(nn.Module):
    # create embedder and objects
    def __init__(self,
                 state_size = 4, state_hn = 4, state_embed_size=16,
                 obj_size = 3, obj_n = 3, nullobj_n = 4, obj_hn = 4, obj_embed_size=16,
                 gru_hn=2, gru_out_size=16,
                 rel_hn = 2, rel_out_size = 8,
                 fin_mlp_hn = 2, num_gaussians=2,
                 mlp_hsize=16, mlp_hn=3, mlp_outsize=1
                 ):
        super(relRnn, self).__init__()

        # state embedder
        self.state_size = state_size
        self.state_embed_size = state_embed_size
        self.state_hn = state_hn

        self.state_embedder = MLP(input_size=state_size,
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

        # feed all embeddings into gru
        self.gru_hn = gru_hn
        self.gru_out_size = gru_out_size
        self.register_parameter('init_gru_h',
                                torch.nn.Parameter(torch.rand(gru_hn, gru_out_size)))


        # collision network
        self.mlp_hsize = mlp_hsize
        self.mlp_hn = mlp_hn
        self.mlp_outsize = mlp_outsize
        self.mlp = MLP(input_size = [state_embed_size, obj_embed_size],
                        hidden_layer_size=mlp_hn*[mlp_hsize],
                        output_size=mlp_outsize,
                        activation=F.elu)
        self.scores = PermuteLayer(nullobj_n+obj_n)


        # recurrent network
        self.gru = nn.GRU(input_size= state_embed_size + obj_embed_size*(obj_n + nullobj_n),
                          hidden_size=gru_out_size,
                          num_layers=gru_hn)

        # Add rel net to model respective bounces
        self.rel_hn = rel_hn
        self.rel_out_size = rel_out_size
        self.rel_layer = nn.ModuleList()
        for i in range(obj_n+nullobj_n):
            self.rel_layer.append(nn.ModuleList())
            self.rel_layer[i].append(MLP(input_size=[gru_out_size, obj_embed_size],
                                        hidden_layer_size=rel_hn*[rel_out_size],
                                        output_size=rel_out_size,
                                        activation = F.elu))

        # final aggregator MLP
        self.num_gaussians = num_gaussians
        self.fin_mlp_hn = fin_mlp_hn
        self.out_msize = state_size
        self.out_ssize = int((state_size*state_size+state_size)/2)
        self.output_size = [self.num_gaussians,  # alpha
                            self.num_gaussians * self.out_msize,  # mu
                            self.num_gaussians * self.out_ssize]
        self.final_layer = MLP(input_size = gru_out_size+rel_out_size *(obj_n+nullobj_n),
                                hidden_layer_size=fin_mlp_hn*[sum(self.output_size)],
                                output_size=self.output_size,
                                activation=F.elu)

    def forward(self, envs, states, predict_t=0):
        """
        envs: shape = (batch_size, obj_size*obj_n). Objs in order.
        states: shape = (batch_size, t, state_size)
        if predict_t is 0, only returns predicted GM using true states at each t
        if predict_t > 0, returns GM using true states, GM using sampled states, and samples
        """

        Nbatch = list(envs.shape[:-1])[0]

        # for each object
        obj_embd = []
        for i_obj in range(self.obj_n):
            sidx = self.obj_size*i_obj
            x = envs[:,sidx:(sidx+self.obj_size)]

            # get embedding
            for i_layer in range(len(self.obj_embdder[i_obj])):
                x = self.obj_embdder[i_obj][i_layer](x)
            obj_embd.append(x)

        # for null objects
        for i_nullobj in range(self.nullobj_n):
            x = self.null_embed[i_nullobj,:].view(1,-1).repeat(Nbatch,1)
            obj_embd.append(x)

        # predict max_t future states by sampling from last GM
        gm, h_n, a, m, s = self.feed_all(obj_embd, states)

        if predict_t > 0:
            gm2, samples = self.feed_sequential(obj_embd, gm, h_n, predict_t)
            return gm, gm2, samples
        else:
            return gm

    def feed_all(self, obj_embd, states,h_n = None):
        """
        states: shape = (batch_size, t, state_size )
        obj_embd: list of object embeddiings
            obj embeddings: shape = (batch_size, obj_emb_size)
        At each t, given the true state at t, predict the distribution for t+1

        note: GRU takes inputs of shape (seq_len, batch, input_size); thus all the permutes.
        """
        batch_size, t, state_size = states.shape

        if h_n is None:
            h_n = utils.expand_along_dim(self.init_gru_h, batch_size, 1).contiguous() # shape = (h_n, batch_size, h_size)?

        obj_embd_t = []
        for emb in obj_embd:
            h_env = utils.expand_along_dim(emb, t, 0)
            obj_embd_t.append(h_env)

        states = states.permute(1, 0, 2) #check dimensions
        states_embd = self.state_embedder(states) #embed along the last dimension

        ###############################################################################################################
        # collision network
        scores = []
        for emb in obj_embd_t:
            x = self.mlp(states_embd, emb)
            scores.append(x)

        out = torch.cat(scores, -1)
        col_scores = self.scores(out)
        col_scores = F.softmax(col_scores, dim=-1)
        ###############################################################################################################

        #RNN
        gru_in = torch.cat([states_embd] + obj_embd_t, dim=-1) #shape = ()
        gru_out, h_n = self.gru(gru_in, h_n) #shape = (t, batch, h_size)
        #gru_out = F.sigmoid(gru_out)

        # run output through relational layer
        relnet_scores = []
        for i_obj in range(self.obj_n + self.nullobj_n):
            # get embedding
            for i_layer in range(len(self.rel_layer[i_obj])):
                rel_out = self.rel_layer[i_obj][i_layer](gru_out,obj_embd_t[i_obj]) # shape = (time x batch x size)
                #rel_out = F.sigmoid(rel_out)

            weighted_scores = rel_out * col_scores[...,i_obj].unsqueeze(2).expand_as(rel_out)
            relnet_scores.append(weighted_scores)

        # feed relouts into through mlp
        final_in = torch.cat([gru_out]+relnet_scores, dim=-1)
        a, m, s = self.final_layer(final_in)
        a = a.permute(1, 0, 2)
        m = m.permute(1, 0, 2)
        s = s.permute(1, 0, 2)
        a = F.softmax(a, dim=-1)
        s = F.softplus(s) + epsilon
        m = m.view(a.shape + (self.out_msize,))
        s = s.view(a.shape + (self.out_ssize,))

        gm = GaussianMixture(a, m, s, lower_cholesky=True) #shape = (batch, T, n_gaussian)

        # return gm, and the last states (h_n, a,m,s)
        return gm, h_n, a[:, -1], m[:, -1], s[:, -1]

    def feed_sequential(self,obj_embd, gm, h_n, predict_t):
        """
        init_state: shape = (batch_size,1, state_size)
        At each t, samples a state at t to predict the distribution for t+1
        """
        state = gm.sample() # shape =  (batch x size)

        gms = [gm]
        samples = [state]
        for i in range(predict_t - 1):
            state_embd = self.state_embedder(state.squeeze(1)) # shape =  (batch x size)

            ###############################################################################################################
            # collision network
            scores = []
            for emb in obj_embd:
                x = self.mlp(state_embd, emb)
                scores.append(x)

            out = torch.cat(scores, -1)
            col_scores = self.scores(out)
            col_scores = F.softmax(col_scores, dim=-1)
            ###############################################################################################################

            gru_in = torch.cat([state_embd] + obj_embd, dim=-1).unsqueeze(0) # shape = (1 x batch x size)

            gru_out, h_n = self.gru(gru_in, h_n)
            gru_out = gru_out.squeeze(0)
            #gru_out = F.sigmoid(gru_out)

            relnet_scores = []
            for i_obj in range(self.obj_n + self.nullobj_n):
                # get embedding
                for i_layer in range(len(self.rel_layer[i_obj])):
                    rel_out = self.rel_layer[i_obj][i_layer](gru_out, obj_embd[i_obj]) # shape = batch x size
                    #rel_out = F.sigmoid(rel_out)

                    weighted_scores = rel_out * col_scores[...,i_obj].unsqueeze(-1).expand_as(rel_out)
                relnet_scores.append(weighted_scores)

            # feed relouts into through mlp
            final_in = torch.cat([gru_out] + relnet_scores, dim=-1)
            a, m, s = self.final_layer(final_in)
            a = F.softmax(a, dim=-1)
            s = F.softplus(s) + epsilon
            m = m.view(a.shape + (self.out_msize,))
            s = s.view(a.shape + (self.out_ssize,))

            # sample new position
            gm = GaussianMixture(a, m, s, lower_cholesky=True) #gm shape: (batch x size)
            state = gm.sample().unsqueeze(1)
            samples.append(state)
            gms.append(gm)

        return gms, torch.cat(samples, dim=1)


class relRnn_det(nn.Module):
    # create embedder and objects
    def __init__(self,
                 state_size = 4, state_hn = 4, state_embed_size=16,
                 obj_size = 3, obj_n = 3, nullobj_n = 4, obj_hn = 4, obj_embed_size=16,
                 gru_hn=2, gru_out_size=16,
                 rel_hn = 2, rel_out_size = 8,
                 fin_mlp_hn = 2,
                 mlp_hsize=16, mlp_hn=3, mlp_outsize=1):
        super(relRnn_det, self).__init__()

        # state embedder
        self.state_size = state_size
        self.state_embed_size = state_embed_size
        self.state_hn = state_hn

        self.state_embedder = MLP(input_size=state_size,
                                  hidden_layer_size=state_hn * [state_embed_size],
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
                                           hidden_layer_size=obj_hn * [obj_embed_size],
                                           output_size=obj_embed_size,
                                           activation=F.elu))
            # self.obj_embdder[i].append(MLP(input_size=obj_size,
            #                               hidden_layer_size=obj_hn * [obj_embed_size],
            #                               output_size=obj_embed_size,
            #                               activation=F.elu))

        # null embeddings (i.e. 2 walls, ground), constant across environments
        self.nullobj_n = nullobj_n

        self.register_parameter('null_embed',
                                torch.nn.Parameter(torch.rand(nullobj_n, obj_embed_size)))

        # feed all embeddings into gru
        self.gru_hn = gru_hn
        self.gru_out_size = gru_out_size
        self.register_parameter('init_gru_h',
                                torch.nn.Parameter(torch.rand(gru_hn, gru_out_size)))

        # collision network
        self.mlp_hsize = mlp_hsize
        self.mlp_hn = mlp_hn
        self.mlp_outsize = mlp_outsize
        self.mlp = MLP(input_size=[state_embed_size, obj_embed_size],
                       hidden_layer_size=mlp_hn * [mlp_hsize],
                       output_size=mlp_outsize,
                       activation=F.elu)
        self.scores = PermuteLayer(nullobj_n + obj_n)

        # recurrent network
        self.gru = nn.GRU(input_size=state_embed_size + obj_embed_size * (obj_n + nullobj_n),
                          hidden_size=gru_out_size,
                          num_layers=gru_hn)

        # Add rel net to model respective bounces
        self.rel_hn = rel_hn
        self.rel_out_size = rel_out_size
        self.rel_layer = nn.ModuleList()
        for i in range(obj_n + nullobj_n):
            self.rel_layer.append(nn.ModuleList())
            self.rel_layer[i].append(MLP(input_size=[gru_out_size, obj_embed_size],
                                         hidden_layer_size=rel_hn * [rel_out_size],
                                         output_size=rel_out_size,
                                         activation=F.elu))

        # final aggregator MLP
        self.fin_mlp_hn_det = fin_mlp_hn
        self.final_layer_det = MLP(input_size=gru_out_size + rel_out_size * (obj_n + nullobj_n),
                               hidden_layer_size=fin_mlp_hn * [self.state_size],
                               output_size=self.state_size,
                               activation=F.elu)


    def forward(self, envs, states, predict_t=0):
        """
        envs: shape = (batch_size, obj_size*obj_n). Objs in order.
        states: shape = (batch_size, t, state_size)
        if predict_t is 0, only returns predicted GM using true states at each t
        if predict_t > 0, returns GM using true states, GM using sampled states, and samples
        """

        Nbatch = list(envs.shape[:-1])[0]

        # for each object
        obj_embd = []
        for i_obj in range(self.obj_n):
            sidx = self.obj_size*i_obj
            x = envs[:,sidx:(sidx+self.obj_size)]

            # get embedding
            for i_layer in range(len(self.obj_embdder[i_obj])):
                x = self.obj_embdder[i_obj][i_layer](x)
            obj_embd.append(x)

        # for null objects
        for i_nullobj in range(self.nullobj_n):
            x = self.null_embed[i_nullobj,:].view(1,-1).repeat(Nbatch,1)
            obj_embd.append(x)

        # predict max_t future states by sampling from last GM
        states, h_n = self.feed_all(obj_embd, states)

        if predict_t > 0:
            samples = self.feed_sequential(obj_embd, states, h_n, predict_t)
            return samples
        else:
            return states


    def feed_all(self, obj_embd, states, h_n = None):
        """
        states: shape = (batch_size, t, state_size )
        obj_embd: list of object embeddiings
            obj embeddings: shape = (batch_size, obj_emb_size)
        At each t, given the true state at t, predict the distribution for t+1

        note: GRU takes inputs of shape (seq_len, batch, input_size); thus all the permutes.
        """
        batch_size, t, state_size = states.shape

        if h_n is None:
            h_n = utils.expand_along_dim(self.init_gru_h, batch_size, 1).contiguous() # shape = (h_n, batch_size, h_size)?

        obj_embd_t = []
        for emb in obj_embd:
            h_env = utils.expand_along_dim(emb, t, 0)
            obj_embd_t.append(h_env)

        states = states.permute(1, 0, 2) #check dimensions
        states_embd = self.state_embedder(states) #embed along the last dimension

        ###############################################################################################################
        # collision network
        scores = []
        for emb in obj_embd_t:
            x = self.mlp(states_embd, emb)
            scores.append(x)

        out = torch.cat(scores, -1)
        col_scores = self.scores(out)
        col_scores = F.softmax(col_scores, dim=-1)
        ###############################################################################################################

        gru_in = torch.cat([states_embd] + obj_embd_t, dim=-1) #shape = ()
        gru_out, h_n = self.gru(gru_in, h_n) #shape = (t, batch, h_size)
        #gru_out = F.sigmoid(gru_out)

        # run output through relational layer
        relnet_scores = []
        for i_obj in range(self.obj_n + self.nullobj_n):
            # get embedding
            for i_layer in range(len(self.rel_layer[i_obj])):
                rel_out = self.rel_layer[i_obj][i_layer](gru_out,obj_embd_t[i_obj]) # shape = (time x batch x size)
                #rel_out = F.sigmoid(rel_out) # added this

            weighted_scores = rel_out * col_scores[...,i_obj].unsqueeze(2).expand_as(rel_out)  # weight by collision network output
            relnet_scores.append(weighted_scores)

        # feed relouts into through mlp
        final_in = torch.cat([gru_out]+relnet_scores, dim=-1)
        out = self.final_layer_det(final_in).permute(1, 0, 2)
        return out, h_n

    def feed_sequential(self,obj_embd, states, h_n, predict_t):
        """
        init_state: shape = (batch_size,1, state_size)
        At each t, samples a state at t to predict the distribution for t+1
        """
        samples = [states]
        for i in range(predict_t - 1):
            state_embd = self.state_embedder(states.squeeze(1)) # shape =  (batch x size)

            ###############################################################################################################
            # collision network
            scores = []
            for emb in obj_embd:
                x = self.mlp(state_embd, emb)
                scores.append(x)

            out = torch.cat(scores, -1)
            col_scores = self.scores(out)
            col_scores = F.softmax(col_scores, dim=-1)
            ###############################################################################################################

            gru_in = torch.cat([state_embd] + obj_embd, dim=-1).unsqueeze(0) # shape = (1 x batch x size)
            gru_out, h_n = self.gru(gru_in, h_n)
            gru_out = gru_out.squeeze(0)
            #gru_out = F.sigmoid(gru_out)

            relnet_scores = []
            for i_obj in range(self.obj_n + self.nullobj_n):
                # get embedding
                for i_layer in range(len(self.rel_layer[i_obj])):
                    rel_out = self.rel_layer[i_obj][i_layer](gru_out, obj_embd[i_obj]) # shape = batch x size
                    #rel_out = F.sigmoid(rel_out)

                weighted_scores = rel_out * col_scores[...,i_obj].unsqueeze(-1).expand_as(rel_out)
                relnet_scores.append(weighted_scores)

            # feed relouts into through mlp
            final_in = torch.cat([gru_out] + relnet_scores, dim=-1)
            states = self.final_layer_det(final_in).unsqueeze(1)
            samples.append(states)

            # outputs = []
            # index = 0
            # for size in self.output_size:
            #     outputs.append(out[..., index:index + size])
            #     index += size
            # a, m, s = outputs
            #
            # a = F.softmax(a, dim=-1)
            # s = F.softplus(s) + epsilon
            # m = m.view(a.shape + (2,))
            # s = s.view(a.shape + (3,)) # shape?
            #
            # # sample new position
            # gm = GaussianMixture(a, m, s, lower_cholesky=True) #gm shape: (batch x size)
            # state = gm.sample().unsqueeze(1)
            # samples.append(state)
            # gms.append(gm)

        return torch.cat(samples, dim=1)


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
