import sys
# sys.path.append('../src')
sys.path.append('C:/Users/tepzh/Dropbox/Stanford/CS229 machine learning/project/plinko_nn/src')
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
from plinko.model.predictor_gru import GRUPredictor_mu
from plinko.model.predictor_gru import GRUPredictor_determ
from plinko.misc import plot as plinko_plot


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(0)
torch.manual_seed(0)
torch.set_default_tensor_type('torch.FloatTensor')
epsilon = sys.float_info.epsilon



# repo_path = '..'
repo_path = 'C:/Users/tepzh/Dropbox/Stanford/CS229 machine learning/project/plinko_nn'
df_ball = pd.read_feather(repo_path + '/data/simulations/sim_ball.feather')
df_env = pd.read_feather(repo_path + '/data/simulations/sim_environment.feather')
df_col = pd.read_feather(repo_path + '/data/simulations/sim_collisions.feather')
# df_ball = pd.read_feather(repo_path + '/data/training/sim_ball.feather')
# df_env = pd.read_feather(repo_path + '/data/training/sim_environment.feather')
# df_col = pd.read_feather(repo_path + '/data/training/sim_collisions.feather')

# create df with collision data
collisions = df_col.copy()
collisions = collisions.rename({'object': 'collision'}, axis=1)
collisions.t -= 1
collisions = df_ball.merge(collisions, how='left')
collisions.collision[collisions.collision.isna()] = 'none'
collisions.collision[(collisions.collision == 'walls') & (collisions.px < 100)] = 'left_wall'
collisions.collision[(collisions.collision == 'walls') & (collisions.px > 500)] = 'right_wall'
collisions = collisions.sort_values(['simulation', 'run', 't'])
idx2col = sorted(collisions.collision.unique())
col2idx = {c: i for c, i in zip(idx2col, range(len(idx2col)))}
collisions['col'] = [col2idx[c] for c in collisions.collision]
df_collision = collisions.drop(columns="collision")

sim_data = data_utils.get_sim_data(df_collision, df_col)
selected_runs = sim_data[(sim_data.num_collisions == 2)
                         & (sim_data.duration < 80)
                         & (sim_data.run <= 5)]
simulations, environments = data_utils.create_task_df(selected_runs, df_collision, df_env)
states, envs = data_utils.to_tensors(simulations, environments, device, include_v=True, include_t=True)
states[:, :, 2:4] = (states[:, :, 2:4] + 20) # transform velocity to all positive

# model = GRUPredictor_mu(env_size=11, state_size=2, num_gaussians=1, trainable_h0 = True).to(device)
model = GRUPredictor_determ(env_size=11, state_size=6, gru_hidden_size=64, num_gaussians=1, trainable_h0=True).to(device)
optimizer = optim.Adam(model.parameters(), lr = 10e-4, weight_decay=.001)
dataset = SimulationDataset(envs, states)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)


# Train GRU_deterministic
max_t = simulations.t.max()
epochs = 1000
losses = []
p_overtime = []
v_overtime = []
target_overtime = []
# for epoch in tqdm(range(epochs+1)):
for epoch in tqdm(range(3)):
    epoch_loss = 0
    epoch_p_loss = 0
    epoch_v_loss = 0
    for batch_i, batch in enumerate(dataloader):
        optimizer.zero_grad()

        #         print('Batch ', batch_i)
        p_batch, v_batch, t_batch = model(batch['envs'], batch['states'], 0)
        targets = batch['targets']

        p_mse_loss = F.mse_loss(p_batch, targets[:, :, :2])
        v_mse_loss = .002 * F.mse_loss(v_batch, targets[:, :, 2:4])
        loss = p_mse_loss + v_mse_loss
        loss.backward(retain_graph=True)
        optimizer.step()
        epoch_loss += loss
        epoch_p_loss += p_mse_loss
        epoch_v_loss += v_mse_loss
        losses.append((epoch, batch_i, float(loss)))

        p_overtime.append(p_batch)
        v_overtime.append(v_batch)
        target_overtime.append(targets)

    if epoch % 1 == 0:
        print('Epoch {} | p_loss: {} | v_loss: {} | total: {}'.format(epoch,
                                                                      round(float(epoch_p_loss), 4),
                                                                      round(float(epoch_v_loss), 4),
                                                                      round(float(epoch_loss), 4)))

#     if (loss - prev_loss) < .1 and (loss - prev_loss) > -.1:
#         break

# torch.save(model.state_dict(), 'gru.model')
# torch.save(mu_overtime, 'mu_overtime.pt')
# torch.save(sigma_overtime, 'sigma_overtime.pt')
# torch.save(target_overtime, 'target_overtime.pt')
# torch.save(losses, 'losses.pt')


def simulate_model(model, dataset, sim_t=1):
    """
    :sim_t = how many time points to feed in for the simulation
    """
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    i = 0
    for batch in dataloader:
        i += 1
        with torch.no_grad():
            p, v = model(batch['envs'], batch['states'][:, 0:sim_t], dataset[0]['states'].shape[0] - sim_t)
            targets = batch['targets'][:, 1:101]
            #             df_env, df_ball = data_utils.create_simdata_from_samples(samples, batch['envs'],sim_df, env_df)

            return p, v, targets

sim_p, sim_v, sim_targets = simulate_model(model, dataset, sim_t = 1)

plinko_plot.plot_pred_target(sim_p, sim_targets[:, :, :2], sim_range=range(5),
                             title = "Full simulation: prediction vs. target")