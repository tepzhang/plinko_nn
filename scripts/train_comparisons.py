import os
import sys
sys.path.append('../src')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from plinko.misc import data_utils
from plinko.misc.simulation_dataset import SimulationDataset
from plinko.model.base_mlp import baseMLP
from plinko.model.base_gru import GRUPredictor
from plinko.model.relNetgru import relNetgru
from plinko.model.relRnn_scores import relRnn
from plinko.misc import plot as plinko_plot
# from plotnine import *

import matplotlib.pyplot as plt


def loaddata(maxsetsize = 500, num_colls = None, simdur = 100, run_indices = range(20), outdf = False, basedir = '../data/simulations/',device = 'cpu',outv = False):
    df_ball = pd.read_feather(os.path.join(basedir + 'sim_ball.feather'))
    df_env = pd.read_feather(os.path.join(basedir + 'sim_environment.feather'))
    df_col = pd.read_feather(os.path.join(basedir + 'sim_collisions.feather'))

    sim_data = data_utils.get_sim_data(df_ball, df_col)

    # change this to dur <940 (max, 15s fall)
    if num_colls is None:
        num_colls = range(max(sim_data.num_collisions))

    if simdur is None:
        simdur = max(sim_data.duration)

    selected_runs = sim_data[np.in1d(sim_data.num_collisions,num_colls)
                              & (sim_data.duration < simdur)
                              & np.in1d(sim_data.run,run_indices)]
    # selected_runs = sim_data[(sim_data.num_collisions == 2)
    #                          & (sim_data.duration < 50)
    #                          & np.in1d(sim_data.run,run_indices)]

    # if too big, cut the end
    sizelim = maxsetsize
    if selected_runs.shape[0] > sizelim:
        selected_runs =  selected_runs.sample(sizelim)

    simulations, environments = data_utils.create_task_df(selected_runs, df_ball, df_env, append_t0 = False)

    if outdf:
        states, envs, simulations, environments = data_utils.to_tensors(simulations, environments, device, outdf,outv)
    else:
        states, envs = data_utils.to_tensors(simulations, environments, device, outdf, outv)
    return states, envs, simulations, environments

def get_logp_loss(gm, targets):
    return -gm.log_p(targets).mean()

def get_mu_mse_loss(gm, targets):
    return F.mse_loss(gm.mu[:,:,0], targets)

def train_model(model,optimizer,traindatadir='../data/training/',epochs = 1000, savename = 'gru', dtmnstic_out=False):
    cv_states, cv_envs, cv_sim_df, cv_env_df = loaddata(num_colls=3, simdur=None,
                                                        run_indices=range(99, 100),
                                                        outdf=True, basedir='../data/training/',outv = True)
    cv_set = SimulationDataset(cv_envs.to(device), cv_states.to(device),outv = True)

    losses = []
    allepoch_losses = []
    for epoch in tqdm(range(epochs + 1)):
        epoch_loss = 0
        epoch_mse_loss = 0
        epoch_logp_loss = 0

        runindices = np.random.randint(0,80,1) #select a random index

        torch.cuda.empty_cache()
        train_states, train_envs, train_sim_df, train_env_df = loaddata(maxsetsize = 500, num_colls=None, simdur=150, run_indices=runindices,
                                                                        outdf=True,basedir=traindatadir, device = device,outv = True)  # range(80)
        train_set = SimulationDataset(train_envs, train_states,outv = True)

        # run SGD, with batchsize =64
        dataloader = DataLoader(train_set, batch_size=125, shuffle=True)

        for batch_i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            targets = batch['targets']
            if dtmnstic_out:
                out = model(batch['envs'], batch['states'], 0)
                mse_loss = F.mse_loss(out, targets)
                logp_loss = 0
                loss = mse_loss
            else:
                gm = model(batch['envs'], batch['states'], 0)

                logp_loss = get_logp_loss(gm, targets)
                loss = logp_loss
                mse_loss = 10 * get_mu_mse_loss(gm, targets)

            #loss = logp_loss + mse_loss
            loss.backward(retain_graph=True)
            optimizer.step()

            epoch_loss += loss
            epoch_logp_loss += logp_loss
            epoch_mse_loss += mse_loss
            losses.append((epoch, batch_i, float(loss)))

        if epoch % 10 == 0:
            if dtmnstic_out:
                loss = test_model_mseloss(model, cv_sim_df, cv_set)
                simulate_deterministic_model(model, train_set, train_envs, nsim=1, modelname=savename + '_train' + 'epoch' + str(epoch))
                simulate_deterministic_model(model, cv_set, cv_envs, nsim=1, modelname=savename + '_cv' + 'epoch' + str(epoch))
            else:
                loss, logp_loss, mse_loss = test_model_loglike(model, cv_sim_df, cv_set)
                simulate_model(model, train_set, train_envs, nsim=1, modelname=savename + '_train' + 'epoch' + str(epoch))
                simulate_model(model, cv_set, cv_envs, nsim = 1, modelname=savename + '_cv' + 'epoch'+ str(epoch))

            print('Epoch {} | nlogp: {} | mse: {} | total: {} | cv loss: {}'.format(epoch,
                                                                    round(float(epoch_logp_loss), 4),
                                                                    round(float(epoch_mse_loss), 4),
                                                                    round(float(epoch_loss), 4),
                                                                    round(float(loss.item()), 4)))
            torch.save(model.state_dict(), savename + '_epoch'+ str(epoch) + '.model')
        allepoch_losses.append((epoch, float(epoch_loss)))

    torch.save(model.state_dict(), savename + '.model')
    plinko_plot.plot_trainlosses(allepoch_losses, title='training loss', filename = (savename + '_trainloss'))
    return model

def test_model_mseloss(model, sim_df, dataset):
    # run SGD, with batchsize =64
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch_i, batch in enumerate(dataloader):
        out = model(batch['envs'], batch['states'], 0)
        targets = batch['targets']
        mse_loss = F.mse_loss(out, targets)
    return mse_loss

def test_model_loglike(model,sim_df, dataset):
    # run SGD, with batchsize =64
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    epoch_loss = 0
    epoch_mse_loss = 0
    epoch_logp_loss = 0
    for batch_i, batch in enumerate(dataloader):
        gm = model(batch['envs'], batch['states'], 0)
        targets = batch['targets']
        logp_loss = get_logp_loss(gm, targets)
        loss = logp_loss
        mse_loss = 10 * get_mu_mse_loss(gm, targets)
        #loss = logp_loss + mse_loss
        epoch_loss += loss
        epoch_logp_loss += logp_loss
        epoch_mse_loss += mse_loss

    return epoch_loss, epoch_logp_loss, epoch_mse_loss

def simulate_model(model,dataset, env,nsim = 4,modelname = 'gru1'):
    # load and plot 4 times
    dataloader = DataLoader(dataset, batch_size=20, shuffle=False)

    for batch_i, batch in enumerate(dataloader):
        if batch_i < nsim:
            gm = model(batch['envs'], batch['states'], 0)
            targets = batch['targets']

            # plot mu's

            # plot simulation x_{t+1}|x_t
            gm_mu = gm.sample()
            target_mu = targets
            plinko_plot.plot_pred_target(gm_mu[...,:2], target_mu[...,:2], range(1),filename=(modelname + 'nextsim{}'.format(batch_i)))

            # plot entire simulation
            gm, ex_gm2, samples = model(batch['envs'], batch['states'][:, :1], 100)
            targets = batch['targets'][:,1:101]
            plinko_plot.plot_pred_target(samples[...,:2], targets[...,:2], range(1),filename=(modelname + 'fullsim{}'.format(batch_i)))

            plinko_plot.plot_pred_sim_target(gm_mu[...,:2], samples[...,:2], targets[...,:2], env, sim_range=range(1), env_index=0,
                                 alpha=.5, title="Prediction vs. simulation vs. target", filename=(modelname + 'sim{}'.format(batch_i)))

def simulate_deterministic_model(model,dataset, env,nsim = 4,modelname = 'gru1'):
    # load and plot 4 times
    dataloader = DataLoader(dataset, batch_size=20, shuffle=False)

    for batch_i, batch in enumerate(dataloader):
        if batch_i < nsim:
            nextsim = model(batch['envs'], batch['states'], 0)
            targets = batch['targets']

            # plot simulation x_{t+1}|x_t
            plinko_plot.plot_pred_target(nextsim[...,:2], targets[...,:2], range(1),filename=(modelname + 'nextsim{}'.format(batch_i)))

            # plot entire simulation
            fullsim = model(batch['envs'], batch['states'][:, :1], 100)
            targets = batch['targets'][:,1:101]
            plinko_plot.plot_pred_target(fullsim[...,:2], targets[...,:2], range(1),filename=(modelname + 'fullsim{}'.format(batch_i)))

            plinko_plot.plot_pred_sim_target(nextsim[...,:2], fullsim[...,:2], targets[...,:2], env, sim_range=range(1), env_index=0,
                                 alpha=.5, title="Prediction vs. simulation vs. target", filename=(modelname + 'sim{}'.format(batch_i)))

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_default_tensor_type('torch.FloatTensor')
    epsilon = sys.float_info.epsilon

    savedir = '../experiments/baselines/'
    ensure_dir(savedir)

    train_mlp = False
    train_GRU = False
    train_relGru = False
    train_relRnn = True

    ###################################################################################################################
    # train base_mlp
    ###################################################################################################################
    if train_mlp:
        name = 'baseMLP'
        savename = (savedir + name)

        model = baseMLP(state_size=4, env_size=9,
                        h_size = 144, h_n = 10,
                        num_gaussians=2).to(device)

        # if there is a trained model, load it
        if os.path.isfile(savename + '.model'):
            pretrained_dict = torch.load(savename + '.model')
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)  # 2. overwrite entries in the existing state dict
            model.load_state_dict(model_dict)

        weights, biases = [], []
        for name, p in model.named_parameters():
            if 'bias' in name:
                biases += [p]
            elif 'init_gru_h' in name:
                biases += [p]
            else:
                weights += [p]

        optimizer = optim.Adam([{'params': weights, ' weight_decay':0.001},{'params': biases, 'weight_decay':0}], lr=0.001)
        model = train_model(model, optimizer, traindatadir='../data/training/', epochs=20, savename=savename)

    ###################################################################################################################
    # train GRU
    ###################################################################################################################
    if train_GRU:
        name = 'baseGRU'
        savename = (savedir + name)

        model = GRUPredictor(state_size=4, env_size=9,
                             state_embed_size=16,state_hn_size=4,
                             env_embed_size=128, env_hn_size = 4,
                             num_rnn=2,rnn_hsize = 16,
                             mlp_hn = 6, num_gaussians=2,
                             trainable_h0=True).to(device)

        # if there is a trained model, load it
        if os.path.isfile(savename + '.model'):
            pretrained_dict = torch.load(savename + '.model')
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)  # 2. overwrite entries in the existing state dict
            model.load_state_dict(model_dict)

        weights, biases = [], []
        for name, p in model.named_parameters():
            if 'bias' in name:
                biases += [p]
            elif 'init_gru_h' in name:
                biases += [p]
            else:
                weights += [p]

        optimizer = optim.Adam([{'params': weights, ' weight_decay':0.001},{'params': biases, 'weight_decay':0}], lr=0.001)
        model = train_model(model, optimizer, traindatadir='../data/training/', epochs=20, savename=savename)

    ###################################################################################################################
    # train relGru
    ###################################################################################################################
    if train_relGru:
        name = 'baseRelNet'
        savename = (savedir + name)

        model = relNetgru(state_size=4, state_hn = 4, state_embed_size=16,
                          obj_size = 3, obj_n = 3, nullobj_n = 4, obj_hn = 4, obj_embed_size=16,
                          gru_hn=2, gru_out_size=16,
                          rel_hn = 2, rel_out_size = 8,
                          mlp_hn = 2, num_gaussians=2).to(device)

        # if there is a trained model, load it
        if os.path.isfile(savename + '.model'):
            pretrained_dict = torch.load(savename + '.model')
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)  # 2. overwrite entries in the existing state dict
            model.load_state_dict(model_dict)

        weights, biases = [], []
        for name, p in model.named_parameters():
            if 'bias' in name:
                biases += [p]
            elif 'init_gru_h' in name:
                biases += [p]
            else:
                weights += [p]

        optimizer = optim.Adam([{'params': weights, ' weight_decay':0.001},{'params': biases, 'weight_decay':0}], lr=0.001)
        model = train_model(model, optimizer, traindatadir='../data/training/', epochs=20, savename=savename)

    ###################################################################################################################
    # train relGru
    ###################################################################################################################
    if train_relRnn:
        name = 'baseRelRnn'
        savename = (savedir + name)

        model = relRnn(state_size = 4, state_hn = 4, state_embed_size=16,
                       obj_size = 3, obj_n = 3, nullobj_n = 4, obj_hn = 4, obj_embed_size=16,
                       gru_hn=2, gru_out_size=16,
                       rel_hn = 2, rel_out_size = 8,
                       fin_mlp_hn = 2, num_gaussians=2,
                       mlp_hsize=16, mlp_hn=3, mlp_outsize=1).to(device)

        # if there is a trained model, load it
        if os.path.isfile(savename + '.model'):
            pretrained_dict = torch.load(savename + '.model')
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)  # 2. overwrite entries in the existing state dict
            model.load_state_dict(model_dict)

        weights, biases = [], []
        for name, p in model.named_parameters():
            if 'bias' in name:
                biases += [p]
            elif 'init_gru_h' in name:
                biases += [p]
            else:
                weights += [p]

        optimizer = optim.Adam([{'params': weights, ' weight_decay':0.001},{'params': biases, 'weight_decay':0}], lr=0.001)
        model = train_model(model, optimizer, traindatadir='../data/training/', epochs=20, savename=savename)