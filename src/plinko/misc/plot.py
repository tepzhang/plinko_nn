import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *
import torch
from . import utils
from . import data_utils
import math


def plot_pred_target(prediction, target, sim_range=range(9),filename=None):
    """
    :prediction: a tensor of predicted positions
    :target: a tensor of target
    :sim_range: index numbers of simulation to plot
    :return: plot the path of target and prediction
    """

    df = pd.DataFrame()
    for i in sim_range:
        df_pred = pd.DataFrame(prediction[i,:,:2].data.cpu().numpy(),columns = ['px', 'py'])
        df_pred['source'] = 'pred'
        df_target = pd.DataFrame(target[i].data.cpu().numpy(),columns = ['px', 'py'])
        df_target['source'] = 'target'
        df_combined = df_pred.append(df_target)
        df_combined['run'] = str(i)
        df = df.append(df_combined)

    p = (ggplot(df, aes('px', 'py', color = 'source', grouping = 'run'))
        + geom_path(alpha = .2)
        +geom_point(alpha = .2)
        +xlim(0, 10)
        +ylim(0, 10))

    if filename is not None:
        s = filename.split('/')
        path = '/'.join(s[:-1])
        ggsave(filename = s[-1] + '.png',
               path = path,
               plot = p,
               device = 'png',
               dpi = 300, limitsize = True)
    else:
        print(p)

def plot_losses(losses, time_range = None, title = 'loss over time',filename=None):
    """
    :losses: tuples or array of losses (each row is one timepoint, and the 3 columns are epoch, batch_i, and loss)
    :time_range: the range in timepoint you want to plot
    :title: title of the plot
    :return: plot the loss over time
    """
    if time_range is None:
        time_range = range(len(losses))
    df_losses = pd.DataFrame(losses, columns =['epoch', 'batch_i', 'loss'])
    df_losses['time'] = range(len(df_losses))
    # print(df_losses)

    p = (ggplot(df_losses, aes('time', 'loss'))
        + geom_path()
        + geom_point()
        + xlim(time_range[0], time_range[-1])
        + labs(title = title)
        )

    if filename is not None:
        s = filename.split('/')
        path = '/'.join(s[:-1])
        ggsave(filename = s[-1] + '.png',
               path = path,
               plot = p,
               device = 'png',
               dpi = 300, limitsize = True)
    else:
        print(p)

def plot_mu_over_time(mu_overtime, sim_range = None, time_range = None,filename=None):
    """
    :mu_overtime: a list of tensors of mu over time (an element in the list represents on timepoint, each tensor represents one simulation run)
    :time_range: the range in timepoint you want to plot
    :sim_range: index number of simulation to plot
    :return: plot positions (x and y) of average mu in multiple worlds over time
    """
    if time_range is None:
        time_range = range(len(mu_overtime))
    if sim_range is None:
        sim_range = range(mu_overtime[0].shape[0])
    
    df = pd.DataFrame()
    for t in time_range:
        df_world = pd.DataFrame()
        mu = mu_overtime[t]
        for i in sim_range:
            df_mu = pd.DataFrame(mu[i].data.cpu().numpy(), columns = ['px', 'py'])
            df_mu_mean = pd.DataFrame({'x': [df_mu['px'].mean()], 'y': [df_mu['py'].mean()], 'world': [i]})
            df_world = df_world.append(df_mu_mean, ignore_index = True)
    #     print(df_world)
        df = df.append([[df_world['x'].mean(), df_world['y'].mean(), t]], ignore_index = True)
    df = df.rename(columns={0: "x", 1: "y", 2: "time"})

    df = pd.melt(df, id_vars=['time'], value_vars=['x', 'y'], var_name='direction', value_name='coordinate')
    # print(df)

    p = (ggplot(df, aes('time', 'coordinate', color = 'direction'))
        + geom_path()
        + geom_point()
        # + xlim(time_range[0], time_range[-1])
        + labs(title = 'Average mu over time', x = 'time of training')
        )

    if filename is not None:
        s = filename.split('/')
        path = '/'.join(s[:-1])
        ggsave(filename = s[-1] + '.png',
               path = path,
               plot = p,
               device = 'png',
               dpi = 300, limitsize = True)
    else:
        print(p)

def plot_variance_over_time(sigma_overtime, sim_range= None, time_range = None,filename=None):
    """
    :sigma_overtime: a list of tensors of sigma over time (an element in the list represents on timepoint, each tensor represents one simulation run)
    :time_range: the range in timepoint you want to plot
    :sim_range: index number of simulation to plot
    :return: plot average variance (in x and y directions) in multiple worlds over time
    """
    if time_range is None:
        time_range = range(len(sigma_overtime))
    if sim_range is None:
        sim_range = range(sigma_overtime[0].shape[0])

    df = pd.DataFrame()
    for t in time_range:
        sigma = sigma_overtime[t]
        df_sigma = []
        for i in sim_range:
            for j in range(len(sigma[i])):
                sigma_ij = sigma[i, j].data.cpu().numpy()
    #             print(sigma_ij)
                var_x = sigma_ij[0, 0]
                var_y = sigma_ij[1, 1]
    #             print(var_x, var_y)
            df_sigma.append([var_x, var_y])
        df_sigma = pd.DataFrame(df_sigma)
    #     print(df_sigma)
        df = df.append([[df_sigma[0].mean(), df_sigma[1].mean(), t]], ignore_index = True)
    df = df.rename(columns={0: "x", 1: "y", 2: "time"})

    df = pd.melt(df, id_vars=['time'], value_vars=['x', 'y'], var_name='direction', value_name='variance')
    # print(df)

    p = (ggplot(df, aes('time', 'variance', color = 'direction'))
        + geom_path()
        + geom_point()
    #     + xlim(time_range[0], time_range[-1])
        + labs(title = 'Average variance over time', x = 'time of training')
        )

    if filename is not None:
        s = filename.split('/')
        path = '/'.join(s[:-1])
        ggsave(filename = s[-1] + '.png',
               path = path,
               plot = p,
               device = 'png',
               dpi = 300, limitsize = True)
    else:
        print(p)

def plot_pred_sim_target(prediction, simulation, target, env, sim_range=range(10), env_index = 0,
                        alpha = .5, title = "Prediction vs. simulation vs. target", filename=None):
    """
    :prediction: a tensor list of predicted positions
    :simulation: a tensor list of simulation results
    :target: a tensor list of target
    :env: a tensor list containing the environment information in (x, y, r)
    :sim_range: index numbers of simulation to plot
    :env_index: index of which world to plot
    :return: plot the path of target and prediction
    """
    df_env_single = pd.DataFrame(env[env_index].data.cpu().numpy()[:9].reshape(1, 9), columns = ['triangle_x', 'triangle_y',
        'triangle_r', 'rectangle_x', 'rectangle_y', 'rectangle_r', 'pentagon_x',
        'pentagon_y', 'pentagon_r'])
    df_env_trans = pd.DataFrame()
    for col in df_env_single.columns:
        if col[-1] == 'x' or col[-1] == 'y':
            df_env_trans[col] = df_env_single[col] * 70
        else:
            df_env_trans[col] = df_env_single[col] * 2 * math.pi / 10
    df_env_trans['simulation'] = 'sim_' + str(env_index)
    df_env_shapes = data_utils.make_shape_df(df_env_trans)
    df_env_shapes
    df_env_shapes_long = pd.melt(df_env_shapes, id_vars='simulation')
    new = df_env_shapes_long["variable"].str.split("_", n = 2, expand = True)
    df_env_shapes_long["shape"]= new[0]
    df_env_shapes_long["coordinates"]= new[1].str[:2]
    df_env_shapes_long["number"]= df_env_shapes_long["variable"].str[-1:]
    df_env_shapes_long = df_env_shapes_long.drop(columns = "variable")
    df_env_shapes_long = df_env_shapes_long.pivot_table(index=['simulation', 'shape', 'number'], columns='coordinates',values='value').reset_index()
    df_env_shapes_long = df_env_shapes_long.reset_index()


    df = pd.DataFrame()
    for i in sim_range:
        df_pred = pd.DataFrame(prediction[i].data.cpu().numpy(),columns = ['px', 'py'])
        df_pred['source'] = 'pred'
        df_sim = pd.DataFrame(simulation[i].data.cpu().numpy(),columns = ['px', 'py'])
        df_sim['source'] = 'sim'
        df_target = pd.DataFrame(target[i].data.cpu().numpy(),columns = ['px', 'py'])
        df_target['source'] = 'target'
        df_combined = df_pred.append(df_sim)
        df_combined = df_combined.append(df_target)
        df_combined['run'] = str(i)
        df = df.append(df_combined)
    # revert back to the original size
    df['px'] *= 70
    df['py'] *= 70

    p = (ggplot(df, aes('px', 'py', color = 'source', grouping = 'run'))
        + geom_polygon(df_env_shapes_long, aes('vx', 'vy', fill = 'shape'), color = 'black', inherit_aes = False)
        + geom_path(alpha = alpha)
        # +geom_point(alpha = alpha)
        + scale_x_continuous(limits = [0, 700], expand = [0, 0])
        + scale_y_continuous(limits = [0, 650], expand = [0, 0])
        + labs(title = title, x = 'x', y = 'y'))

    if filename is not None:
        s = filename.split('/')
        path = '/'.join(s[:-1])
        ggsave(filename = s[-1] + '.png',
               path = path,
               plot = p,
               device = 'png',
               dpi = 300, limitsize = True)
    else:
        print(p)

def plot_trainlosses(losses, title = 'training loss',filename=None):
    """
    :losses: tuples or array of losses (epoch, loss)
    :title: title of the plot
    :return: plot the loss over time
    """

    df_losses = pd.DataFrame(losses, columns =['epoch', 'loss'])
    # print(df_losses)

    p = (ggplot(df_losses, aes('epoch', 'loss'))
        + geom_path()
        + geom_point()
        + labs(title = title)
        )

    if filename is not None:
        s = filename.split('/')
        path = '/'.join(s[:-1])
        ggsave(filename = s[-1] + '.png',
               path = path,
               plot = p,
               device = 'png',
               dpi = 300, limitsize = True)
    else:
        print(p)

