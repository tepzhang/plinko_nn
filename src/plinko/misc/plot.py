import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *


def plot_pred_target(prediction, target, sim_index):
    """
    :prediction: a tensor of predicted positions
    :target: a tensor of target
    :sim_index: index number of simulation to plot
    :return: print the path of target and prediction
    """
    
    df_pred = pd.DataFrame(prediction.data[sim_index].cpu().numpy(),columns = ['px', 'py'])
    df_pred['source'] = 'pred'
    df_target = pd.DataFrame(target.data[sim_index].cpu().numpy(),columns = ['px', 'py'])
    df_target['source'] = 'target'
    df_combined = df_pred.append(df_target)
#         print(df_combined)

    p = (ggplot(df_combined, aes('px', 'py', group = 'source', color = 'source'))
    + geom_path(alpha = .5)
     +geom_point(alpha = .5)
        +xlim(4, 6))
    print(p)