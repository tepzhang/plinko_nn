import os
import sys
sys.path.append('../../')

from plinko_nn.src.plinko.misc import datamaker

if __name__ == '__main__':
    [df_env, df_ball, df_col] = datamaker.create_sim_data(1000, runs=100)
    df_ball.to_feather(os.path.join('../data/training/' + 'sim_ball.feather'))
    df_col.to_feather(os.path.join('../data/training/' + 'sim_collisions.feather'))
    df_env.to_feather(os.path.join('../data/training/' + 'sim_environment.feather'))


def check_data(model,dataset,sim_df, env_df,nsim = 4,modelname = 'gru1'):
    # load and plot 4 times
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch_i, batch in enumerate(dataloader):
        if batch_i < nsim:
            gm = model(batch['envs'], batch['states'], 0)
            targets = batch['targets']

            # plot mu's


            # plot simulation x_{t+1}|x_t
            gm_mu = gm.mu[:, :, 0]
            target_mu = targets
            plinko_plot.plot_pred_target(gm_mu, target_mu, range(1),filename=(modelname + 'nextsim{}'.format(batch_i)))

            # plot entire
            inter_gm, extra_gm, samples = model(batch['envs'], batch['states'][:, :1], 100)
            targets = batch['targets'][:,1:101]

            plinko_plot.plot_pred_target(samples, targets, range(1),filename=(modelname + 'fullsim{}'.format(batch_i)))
