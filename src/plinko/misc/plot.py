import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *


def plot_pred_target(prediction, target, sim_range=range(9)):
    """
    :prediction: a tensor of predicted positions
    :target: a tensor of target
    :sim_index: index number of simulation to plot
    :return: print the path of target and prediction
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
        +xlim(0, 10))
    print(p)


def plot_losses(losses, time_range = range(len(losses)), title = 'loss over time'):
    """
    :losses: tuples or array of losses (each row is one timepoint, and the 3 columns are epoch, batch_i, and loss)
    :title: title of the plot
    :return: plot the loss over time
    """
    df_losses = pd.DataFrame(losses, columns =['epoch', 'batch_i', 'loss'])
    df_losses['time'] = range(len(df_losses))
    # print(df_losses)

    p = (ggplot(df_losses, aes('time', 'loss'))
        + geom_path()
        + geom_point()
        + xlim(time_range[0], time_range[-1])
        + labs(title = title)
        )
    print(p)