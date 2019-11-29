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
from plinko.model.predictor_gru import GRUPredictor
from plinko.misc import plot as plinko_plot
from plotnine import *

import matplotlib.pyplot as plt


def loaddata(run_indices = range(20), outdf = False, basedir = '../data/simulations/'):
    df_ball = pd.read_feather(os.path.join(basedir + 'sim_ball.feather'))
    df_env = pd.read_feather(os.path.join(basedir + 'sim_environment.feather'))
    df_col = pd.read_feather(os.path.join(basedir + 'sim_collisions.feather'))

    sim_data = data_utils.get_sim_data(df_ball, df_col)

    # change this to dur <940 (max, 15s fall)
    selected_runs = sim_data[(sim_data.num_collisions == 1)
                             & (sim_data.duration < 50)
                             & np.in1d(sim_data.run,run_indices)]
    simulations, environments = data_utils.create_task_df(selected_runs, df_ball, df_env, append_t0 = False)
    if outdf:
        states, envs, simulations, environments = data_utils.to_tensors(simulations, environments, device, outdf)
    else:
        states, envs = data_utils.to_tensors(simulations, environments, device, outdf)
    return states, envs, simulations, environments

def get_logp_loss(gm, targets):
    return -gm.log_p(targets).mean()

def get_mu_mse_loss(gm, targets):
    return F.mse_loss(gm.mu[:,:,0], targets)

def train_model(model,optimizer,simulations,dataset,epochs = 1000, savename = 'gru'):
    # run SGD, with batchsize =64
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    max_t = simulations.t.max()
    losses = []
    allepoch_losses = []
    for epoch in tqdm(range(epochs + 1)):
        epoch_loss = 0
        epoch_mse_loss = 0
        epoch_logp_loss = 0
        for batch_i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            gm = model(batch['envs'], batch['states'], 0)
            targets = batch['targets']

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
        if epoch % 50 == 0:
            print('Epoch {} | logp: {} | mse: {} | total: {}'.format(epoch,
                                                                     round(float(epoch_logp_loss), 4),
                                                                     round(float(epoch_mse_loss), 4),
                                                                     round(float(epoch_loss), 4)))
        allepoch_losses.append((epoch,float(epoch_loss)))
    torch.save(model.state_dict(), savename + '.model')

    plinko_plot.plot_trainlosses(allepoch_losses, title='training loss', filename = (savename + '_trainloss'))
    return model


def test_model_loglike(model,sim_df, dataset, savename = 'gru'):
    # run SGD, with batchsize =64
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    max_t = sim_df.t.max()
    losses = []
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

def simulate_model(model,dataset,sim_df, env_df,nsim = 4,modelname = 'gru1'):
    # load and plot 4 times
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch_i, batch in enumerate(dataloader):
        if batch_i < nsim:
            gm = model(batch['envs'], batch['states'], 0)
            targets = batch['targets']

            # plot simulation x_{t+1}|x_t
            gm_mu = gm.mu[:, :, 0]
            target_mu = targets
            plinko_plot.plot_pred_target(gm_mu, target_mu, range(1),filename=(modelname + 'nextsim{}'.format(batch_i)))

            # plot entire
            inter_gm, extra_gm, samples = model(batch['envs'], batch['states'][:, :1], 100)
            targets = batch['targets'][:,1:101]

            plinko_plot.plot_pred_target(samples, targets, range(1),filename=(modelname + 'fullsim{}'.format(batch_i)))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_default_tensor_type('torch.FloatTensor')
    epsilon = sys.float_info.epsilon

    ## load data
    # range(10)
    test_states, test_envs, test_sim_df, test_env_df = loaddata(run_indices = range(10), outdf = True,basedir = '../data/simulations/')
    test_set = SimulationDataset(test_envs, test_states)

    #range(80)
    train_states, train_envs, train_sim_df, train_env_df = loaddata(run_indices = range(10), outdf = True,basedir = '../data/training/')
    train_set = SimulationDataset(train_envs, train_states)

    cv_states, cv_envs, cv_sim_df, cv_env_df = loaddata(run_indices = range(81,100), outdf = True,basedir = '../data/training/')
    cv_set = SimulationDataset(cv_envs, cv_states)

    # define models and run cross validation
    n_gauss_list = [1,2,4,8,16]
    n_rnn_list  = [1,2,4,8,16]
    reg_list = [0.001, 0.01, 0.1, 1]
    all_cvloglikes = []
    all_names = []
    all_ngauss =[]
    all_nrnn = []
    all_reg = []

    for ngauss in n_gauss_list:
        for nrnn in n_rnn_list:
            for nreg in reg_list:
                name = 'gru_ngauss={}_nrnn={}_reg={}'.format(ngauss,nrnn,nreg)
                all_names.append(name)
                all_ngauss.append(ngauss)
                all_nrnn.append(nrnn)
                all_reg.append(nreg)

                # define model
                model = GRUPredictor(env_size=train_envs.shape[1], state_size=2, num_gaussians=ngauss,num_rnn = nrnn).to(device)

                # train model
                optimizer = optim.Adam(model.parameters(), weight_decay=nreg)
                model = train_model(model, optimizer, train_sim_df, train_set,epochs = 1, savename = ('../experiments/gru_cv/' + name))

                # get loglikelihood and simulate
                with torch.no_grad():
                    loss, logp_loss, mse_loss = test_model_loglike(model, cv_sim_df, cv_set, savename = ('../experiments/gru_cv/' + name))
                    all_cvloglikes.append(loss)
                    simulate_model(model, cv_set, cv_sim_df, cv_env_df, modelname= ('../experiments/gru_cv/' + name))

                plt.close("all")
                torch.cuda.empty_cache()

    bestidx = all_cvloglikes.index(min(all_cvloglikes))
    print('Model with lowest cv error: '.format(all_names[bestidx]))

    cvdata = {'Name': all_names,
            'loglike': all_cvloglikes,
            'N_gaussian': all_ngauss,
            'N_rnn_layer': all_nrnn,
            'Reg weight': all_reg}

    torch.save(pd.DataFrame(cvdata), 'cvmodels')

    # print test error and the training loss
