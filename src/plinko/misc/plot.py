import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *


def plot_pred_target(prediction, target, sim_range=range(9), title = "Prediction (x_t+1|x_t) vs. target"):
    """
    :prediction: a tensor of predicted positions
    :target: a tensor of target
    :sim_range: index numbers of simulation to plot
    :return: plot the path of target and prediction
    """
    df = pd.DataFrame()
    for i in sim_range:
        df_pred = pd.DataFrame(prediction[i].data.cpu().numpy(),columns = ['px', 'py'])
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
        + labs(title = title))
    print(p)

def plot_pred_gaussian(pred_mu, target, sigma, sim_index = 0, title = "95% ellipse of predicted gaussian"):
    """
    pred_mu: a tensor of predicted positions at a certain epoch
    target: a tensor of target at a certain epoch
    sigma: a tensor of predicted sigma at a certain epoch
    sim_index: one index number of simulation to plot
    return: plot the path of target and prediction
    """
    df_pred_mu = pd.DataFrame()
    df_target_mu = pd.DataFrame()
    i = sim_index
    df_pred = pd.DataFrame(pred_mu[i].data.cpu().numpy(),columns = ['px', 'py'])
    df_target = pd.DataFrame(target[i].data.cpu().numpy(),columns = ['px', 'py'])
    df_pred_mu = df_pred_mu.append(df_pred) 
    df_target_mu = df_target_mu.append(df_target)
    # print(df_pred_mu)
    # print(df_target_mu)

    df_sigma = []
    i = sim_index
    for j in range(len(sigma[i])):
        sigma_ij = sigma[i, j].data.cpu().numpy()
        sigma_ij[0, 1] = sigma_ij[1, 0]
        df_sigma.append(sigma_ij)
        
    # print(len(df_sigma))
    sample = pd.DataFrame()
    for i in range(len(df_pred_mu)):
        mean = [df_pred_mu['px'][i], df_pred_mu['py'][i]]
        cov = df_sigma[i]
        x, y = np.random.multivariate_normal(mean, cov, 100).T
        sample = sample.append(pd.DataFrame({'x': x, 'y':y, 'time': i}))
    # print(sample)

    p = (ggplot(sample, aes('x', 'y')) 
        + geom_path(data = df_target_mu, mapping = aes('px', 'py'), alpha = .5, color = 'blue') 
        + geom_point(data = df_target_mu, mapping = aes('px', 'py'), alpha = .5, shape = '.', color = 'blue') 
        + geom_path(data = df_pred_mu, mapping = aes('px', 'py'), alpha = .5) 
        + geom_point(data = df_pred_mu, mapping = aes('px', 'py'), alpha = .5, shape = '.') 
        + stat_ellipse(aes(group = 'time'), alpha = .5)
        + xlim(0, 10)
        + ylim(0, 10)
        + labs(title = title))

    print(p)


def plot_losses(losses, time_range = None, title = 'loss over time', alpha = .5):
    """
    :losses: tuples or array of losses (each row is one timepoint, and the 3 columns are epoch, batch_i, and loss)
    :time_range: the range in timepoint you want to plot
    :title: title of the plot
    :return: plot the loss over time
    """
    df_losses = pd.DataFrame(losses, columns =['epoch', 'batch_i', 'loss'])
    df_losses = df_losses.groupby(['epoch']).sum()
    df_losses['time'] = range(len(df_losses))
    if time_range is None:
        time_range = range(len(df_losses))
    # print(df_losses)

    p = (ggplot(df_losses, aes('time', 'loss'))
        + geom_path(alpha = alpha)
        + geom_point(alpha = alpha)
        + xlim(time_range[0], time_range[-1])
        + labs(title = title)
        )
    print(p)

def plot_mu_over_time(mu_overtime, sim_range = None, time_range = None):
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
        + labs(title = 'Average mu over time', x = 'epoch of training')
        )
    print(p)

def plot_variance_over_time(sigma_overtime, sim_range= None, time_range = None):
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
        + labs(title = 'Average variance over time', x = 'epoch of training')
        )
    print(p)