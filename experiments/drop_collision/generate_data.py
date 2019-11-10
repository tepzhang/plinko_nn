import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from math import pi
import random
import os
from plinko.simulation import config
from plinko.simulation import engine


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n', type=int, default=1000, help='number of simulations')
    parser.add_argument('--t_low', type=int, default=-1, help='timesteps after collision')
    parser.add_argument('--t_high', type=int, default=3, help='timesteps after collision')
    parser.add_argument('--savefile', type=str, default='simulations', help='savefile name (.feather)')

    return parser.parse_args()


def scenario(c):
    """
    Creates a config where the ball is dropped from the center hole onto an obstacle directly below
    :param c:
    :return:
    """
    c['hole_dropped_into'] = 1
    shape = random.choice([('triangle', 3), ('rectangle', 4), ('pentagon', 5)])

    c['obstacles'] = {shape[0]: {'elasticity': 0.0,
                                 'friction': 0.0,
                                 'material': 'wood',
                                 'n_sides': shape[1],
                                 'position': {'x': np.random.randint(340, 360),
                                              'y': np.random.randint(340, 360)},
                                 'rotation': random.uniform(0, 2 * pi),
                                 'size': 50}}
    return c


def create_simulations(n, t_low, t_high):
    """
    Runs n simulations and returns a pandas DataFrame with n rows
    Each row contains the config for the obstacle
        and the position/velocity information for each timestep between
        [t + t_low, t + t_high] where t is when the collision occurred.

    :param n: number of simulations to run
    :param t_low: start of window, can be negative
    :param t_high: end of window, can be negative (but why?)
    :return: pandas DataFrame with n rows
    """
    assert t_high > t_low
    rows = []
    for i in tqdm(range(n)):
        c = scenario(config.get_config())
        s = engine.run_simulation(c)
        t_collision = s['collisions'][0]['step']
        obstacle = list(c['obstacles'].values())[0]

        row = {
            'obs_n_sides': obstacle['n_sides'],
            'obs_x': obstacle['position']['x'],
            'obs_y': obstacle['position']['y'],
            'obs_rot': obstacle['rotation']
        }
        for j in range(t_low, t_high+1):
            row['px_{}'.format(str(j).zfill(2))] = s['ball_position'][t_collision + j - 1]['x']
            row['py_{}'.format(str(j).zfill(2))] = s['ball_position'][t_collision + j - 1]['y']
            row['vx_{}'.format(str(j).zfill(2))] = s['ball_velocity'][t_collision + j - 1]['x']
            row['vy_{}'.format(str(j).zfill(2))] = s['ball_velocity'][t_collision + j - 1]['y']
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    args = parse_args()
    df_collisions = create_simulations(args.n, args.t_low, args.t_high)
    df_collisions.to_feather(os.path.join(args.savefile + '.feather'))
