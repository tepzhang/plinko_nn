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


def loaddata():
    df_ball = pd.read_feather('../data/simulations/sim_ball.feather')
    df_env = pd.read_feather('../data/simulations/sim_environment.feather')
    df_col = pd.read_feather('../data/simulations/sim_collisions.feather')

    sim_data = data_utils.get_sim_data(df_ball, df_col)
    selected_runs = sim_data[(sim_data.num_collisions == 1)
                             & (sim_data.duration < 50)
                             & (sim_data.run <= 2)]
    simulations, environments = data_utils.create_task_df(selected_runs, df_ball, df_env)
    states, envs = data_utils.to_tensors(simulations, environments, device)
    return states, envs, simulations, environments

def get_logp_loss(gm, targets):
    return -gm.log_p(targets).mean()

def get_mu_mse_loss(gm, targets):
    return F.mse_loss(gm.mu[:,:,0], targets)


def train_model(model,optimizer,simulations,dataset):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    max_t = simulations.t.max()
    epochs = 1000
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

    torch.save(model.state_dict(), 'gru.model')
    return model

def simulate_model(model,dataset):
    dataloader = DataLoader(dataset, batch_size=len(envs), shuffle=True)
    i = 0
    for batch in dataloader:
        i += 1
        with torch.no_grad():
            inter_gm, extra_gm, samples = model(batch['envs'], batch['states'][:, :1], 100)
            targets = batch['targets'][:,1:101]



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_default_tensor_type('torch.FloatTensor')
    epsilon = sys.float_info.epsilon

    # load data
    states, envs, simulations, environments = loaddata()
    dataset = SimulationDataset(envs, states)

    # define model
    model = GRUPredictor(env_size=11, state_size=2, num_gaussians=4).to(device)

    # train model;
    # optimizer = optim.SGD(model.parameters(), lr=.001)
    optimizer = optim.Adam(model.parameters(), weight_decay=.001)
    model = train_model(model, optimizer, simulations, dataset)

    # simulate from trained model
    simulate_model(model, dataset)
