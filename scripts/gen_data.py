import os
import sys
sys.path.append('../../')

from plinko_nn.src.plinko.misc import datamaker

if __name__ == '__main__':
    # [df_env, df_ball, df_col] = datamaker.create_sim_data(1000, runs=100)
    # df_ball.to_feather(os.path.join('../data/training/' + 'sim_ball.feather'))
    # df_col.to_feather(os.path.join('../data/training/' + 'sim_collisions.feather'))
    # df_env.to_feather(os.path.join('../data/training/' + 'sim_environment.feather'))
    [df_env, df_ball, df_col] = datamaker.create_sim_data(100, runs=10, no_noise=True)
    df_ball.to_feather(os.path.join('../data/simulations/' + 'sim_ball_no_noise.feather'))
    df_col.to_feather(os.path.join('../data/simulations/' + 'sim_collisions_no_noise.feather'))
    df_env.to_feather(os.path.join('../data/simulations/' + 'sim_environment_no_noise.feather'))
