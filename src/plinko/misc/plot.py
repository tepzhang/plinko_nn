import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *


def plot_pred_target(prediction, target, sim_range):
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
    #         print(df_combined)
        df = df.append(df_combined)
#     print(df)

    p = (ggplot(df, aes('px', 'py', color = 'source', grouping = 'run'))
        + geom_path(alpha = .2)
     +geom_point(alpha = .2)
        +xlim(0, 10))
    print(p)