import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ..simulation import config
from ..simulation import utils
from ..simulation import engine
from . import utils


def load_simulations(sim_directory):
    """
    For loading Tobi's original plinko experiment simulations
    :param sim_directory: directory with the .json files
    :return:
    """
    simulations = {}
    for filename in sorted(Path(sim_directory).rglob('*.json')):
        simname = str(filename).split(sim_directory)[1].split('.json')[0]
        with open(filename) as f:
            simulations[simname] = json.load(f)
    return simulations


def extract_obstacles(obstacles):
    """
    :param obstacles: as from config.get_config()['obstacles']
    :return: flattened dict with obstacle data
    """
    row = {}
    for shape in ('triangle', 'rectangle', 'pentagon'):
        row[shape + '_x'] = obstacles[shape]['position']['x']
        row[shape + '_y'] = obstacles[shape]['position']['y']
        row[shape + '_r'] = obstacles[shape]['rotation']
    return row


def extract_ball_data(simulation):
    """
    :param simulation: as from running engine.run_simulation(config)
    :return: array of dict with ball data
    """
    rows = []
    p_key = 'position' if 'position' in simulation else 'ball_position'
    v_key = 'velocity' if 'position' in simulation else 'ball_velocity'
    for t in range(len(simulation[p_key])):
        row = {
            't': t,
            'px': simulation[p_key][t]['x'],
            'py': simulation[p_key][t]['y'],
            'vx': simulation[v_key][t]['x'],
            'vy': simulation[v_key][t]['y']
        }
        rows.append(row)
    return rows


def extract_collision_data(simulation):
    """
    :param simulation: as from running engine.run_simulation(config)
    :return: array of dict with collision data
    """
    return [{'object': c['objects'][1], 't': c['step']} for c in simulation['collisions']]


def load_env_data(simulations):
    """
    Specifically to be used with Tobi's original plinko simulation json files
    :param simulations: from load_simulations()
    :return: a pandas DataFrame of variable config data
    """
    rows = []
    for sim, data in simulations.items():
        row = {'simulation': sim,
               'hole_dropped_into': data['global']['hole_dropped_into']}
        row = utils.merge_dicts(row, extract_obstacles(data['obstacles']))
        rows.append(row)
    return pd.DataFrame(rows)


def load_ball_data(simulations):
    """
    Specifically to be used with Tobi's original plinko simulation json files
    :param simulations: from load_simulations()
    :return: a pandas DataFrame of ball data
    """
    rows = []
    for sim, data in simulations.items():
        for run in range(len(data['simulation'])):
            ball = data['simulation'][run]['ball']
            row = {
                'simulation': sim,
                'run': run,
            }
            rows += [utils.merge_dicts(row, ball_data) for ball_data in extract_ball_data(ball)]
    return pd.DataFrame(rows)


def load_collision_data(simulations):
    """
    Specifically to be used with Tobi's original plinko simulation json files
    :param simulations: from load_simulations()
    :return: a pandas DataFrame of collision data
    """
    rows = []
    for sim, data in simulations.items():
        for run in range(len(data['simulation'])):
            row = {'simulation': sim, 'run': run}
            rows += [utils.merge_dicts(row, col) for col in extract_collision_data(data['simulation'][run])]
    return pd.DataFrame(rows)


def load_df_sim(sim_directory):
    """
    Specifically to be used with Tobi's original plinko simulation json files
    :param simulations: from load_simulations()
    :return: 3 pandas DataFrames with env, ball, and collision data
    """
    simulations = load_simulations(sim_directory)
    df_env = load_env_data(simulations)
    df_ball = load_ball_data(simulations)
    df_col = load_collision_data(simulations)
    return df_env, df_ball, df_col


def create_sim_data(num_sims, runs=3):
    configs = []
    for i in range(num_sims):
        c = config.get_config()
        c['collision_noise_mean'] = .8
        c['collision_noise_sd'] = .2
        c['drop_noise'] = 0.2
        configs.append(c)

    env_rows = []
    ball_rows = []
    collision_rows = []

    for i in tqdm(range(len(configs))):
        c = configs[i]
        sim_name = 'sim_{}'.format(i)

        # record environment data
        env_row = {'simulation': sim_name, 'hole_dropped_into': c['hole_dropped_into'] + 1}
        env_row = utils.merge_dicts(env_row, extract_obstacles(c['obstacles']))
        env_rows.append(env_row)

        for run in range(runs):
            row = {'simulation': sim_name, 'run': run}
            sim = engine.run_simulation(c)
            ball_rows += [utils.merge_dicts(row, ball_data) for ball_data in extract_ball_data(sim)]
            collision_rows += [utils.merge_dicts(row, col) for col in extract_collision_data(sim)]

    df_env = pd.DataFrame(env_rows)
    df_ball = pd.DataFrame(ball_rows)
    df_col = pd.DataFrame(collision_rows)

    return df_env, df_ball, df_col
