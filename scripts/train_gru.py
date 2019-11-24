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

def loaddata(run_indices = range(20), outdf = False):
    df_ball = pd.read_feather('../data/simulations/sim_ball.feather')
    df_env = pd.read_feather('../data/simulations/sim_environment.feather')
    df_col = pd.read_feather('../data/simulations/sim_collisions.feather')

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


def train_model(model,optimizer,simulations,dataset,savename = 'gru.model'):
    # run SGD, with batchsize =64
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    max_t = simulations.t.max()
    epochs = 1
    losses = []
    for epoch in tqdm(range(epochs + 1)):
        epoch_loss = 0
        epoch_mse_loss = 0
        epoch_logp_loss = 0
        for batch_i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            gm = model(batch['envs'], batch['states'], 0)
            targets = batch['targets']

            logp_loss = get_logp_loss(gm, targets)
            mse_loss = 10 * get_mu_mse_loss(gm, targets)
            loss = logp_loss + mse_loss
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

    torch.save(model.state_dict(), savename)
    return model

def simulate_model(model,dataset,sim_df, env_df,modelname = 'gru1'):
    dataloader = DataLoader(dataset, batch_size=len(envs), shuffle=True)
    i = 0
    for batch in dataloader:
        i += 1
        with torch.no_grad():
            inter_gm, extra_gm, samples = model(batch['envs'], batch['states'][:, :1], 100)
            targets = batch['targets'][:,1:101]
            df_env, df_ball = data_utils.create_simdata_from_samples(samples, batch['envs'],sim_df, env_df)
            df_ball.to_feather(os.path.join('../experiments/' + modelname + '/batch{}'.format(i) + 'samp.feather'))
            df_env.to_feather(os.path.join('../experiments/' + modelname + '/batch_{}'.format(i) + 'envs.feather'))

def todf(env_batch):
    columns = ['' '']


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_default_tensor_type('torch.FloatTensor')
    epsilon = sys.float_info.epsilon

    # load data
    states, envs, sim_df, env_df = loaddata(run_indices = range(20), outdf = True)
    dataset = SimulationDataset(envs, states)

    # define model
    model = GRUPredictor(env_size=envs.shape[1], state_size=2, num_gaussians=4).to(device)

    # train model;
    # optimizer = optim.SGD(model.parameters(), lr=.001)
    optimizer = optim.Adam(model.parameters(), weight_decay=.001)
    model = train_model(model, optimizer,sim_df, dataset)

    # simulate from trained model
    simulate_model(model, dataset,sim_df, env_df)
