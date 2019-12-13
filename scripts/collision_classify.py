repo_path = "D:\ProjGit\plinko\plinko_nn"
import os
import sys
sys.path.append(repo_path + '/src')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal

from plinko.misc import data_utils
from plinko.misc.simulation_dataset import SimulationDataset
from plinko.model.predictor_gru import GRUPredictor
from plinko.misc import plot as plinko_plot
from plinko.model.mlp import MLP
from plinko.model.relNet import relNet
from plinko.misc import utils

from plotnine import * #ggplot, geom_point, aes, labs
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def loaddata(basedir = '../data/simulations/'):
    device_cpu = 'cpu'
    df_ball = pd.read_feather(os.path.join(basedir + 'sim_ball.feather'))
    df_env = pd.read_feather(os.path.join(basedir + 'sim_environment.feather'))
    df_col = pd.read_feather(os.path.join(basedir + 'sim_collisions.feather'))

    return df_ball, df_env, df_col

def createStates(df_ball, df_env, df_col,device):
    collisions = df_col.copy()
    collisions = collisions.rename({'object': 'collision'}, axis=1)
    collisions.t -= 1

    collisions = df_ball.merge(collisions, how='left')
    # collisions = collisions[~collisions.collision.isna()]
    collisions.collision[collisions.collision.isna()] = 'none'
    collisions.collision[(collisions.collision == 'walls') & (collisions.px < 200)] = 'left_wall'
    collisions.collision[(collisions.collision == 'walls') & (collisions.px > 500)] = 'right_wall'
    collisions = collisions.groupby('collision', as_index=False).apply(lambda x: x.sample(2000))
    collisions = collisions.sort_values(['simulation', 'run', 't'])

    env_columns = ['triangle_x', 'triangle_y', 'triangle_r',
                   'rectangle_x', 'rectangle_y', 'rectangle_r',
                   'pentagon_x', 'pentagon_y', 'pentagon_r']
    envs = collisions.merge(df_env).sort_values(['simulation', 'run', 't'])[env_columns]

    idx2col = sorted(collisions.collision.unique())
    col2idx = {c: i for c, i in zip(idx2col, range(len(idx2col)))}

    states = torch.tensor(collisions[['px', 'py', 'vx', 'vy']].values, dtype=torch.float, device=device)
    targets = torch.tensor([col2idx[c] for c in collisions.collision], dtype=torch.long, device=device)
    envs = torch.tensor(envs.values, dtype=torch.float, device=device)

    return states,targets,envs, idx2col, col2idx

def trainModel(envs, states, targets, model,optimizer,epochs,idx2col):
    rows = []
    for epoch in tqdm(range(epochs + 1)):
        optimizer.zero_grad()
        outputs = model(envs, states)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        loss = float(loss)
        corrects = (outputs.argmax(-1) == targets).float()
        accuracy = float(corrects.mean())
        row = {'epoch': epoch, 'loss': loss, 'acc_total': accuracy}
        for idx in range(len(idx2col)):
            row['acc_' + idx2col[idx]] = float(corrects[targets == idx].mean())
        rows.append(row)
        if epoch % 100 == 0:
            print('Epoch {} | loss: {} | acc: {}'.format(epoch,
                                                         round(float(loss), 4),
                                                         round(float(accuracy), 4)))

    summary = pd.DataFrame(rows)

    return summary, model

def testingerror(envs, states, targets, model):
    rows = []
    outputs = model(envs, states)
    loss = F.cross_entropy(outputs, targets)
    loss = float(loss)
    corrects = (outputs.argmax(-1) == targets).float()
    accuracy = float(corrects.mean())

    print("loss = " + str(round(loss, 4)), "acc = " + str(round(accuracy, 4)))
    return loss, accuracy

if __name__ == '__main__':
    repo_path = 'D:\ProjGit\plinko\plinko_nn'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_default_tensor_type('torch.FloatTensor')
    epsilon = sys.float_info.epsilon

    df_ball, df_env, df_col = loaddata(basedir = '../data/simulations/')
    states, targets, envs, idx2col, col2idx = createStates(df_ball, df_env, df_col,device)

    model = relNet(state_size = 4, state_hn = 4, state_embed_size=16,
                   obj_size = 3, obj_n = 3, nullobj_n = 3, obj_hn = 4, obj_embed_size=16,
                   mlp_hsize = 16, mlp_hn=3, mlp_outsize=1,).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    epochs = 7000
    summary = trainModel(envs, states, targets, model,optimizer,epochs,idx2col)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    epochs = 3000
    summary2 = trainModel(envs, states, targets, model,optimizer,epochs,idx2col)

    savename = repo_path + '/experiments/collisions/relnet1'
    torch.save(model.state_dict(), savename + '_9000epochs.model')
    pd.DataFrame(summary).to_pickle(savename + '.pkl')

    # random sample again, to get testing set
    states_test, targets_test, envs_test, idx2col, col2idx = createStates(df_ball, df_env, df_col, device)
    testingerror(envs_test, states_test, targets_test, model)
