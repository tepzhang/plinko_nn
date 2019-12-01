import math
import numpy as np
import torch
import pandas as pd

def get_sim_data(df_ball, df_col):
    """
    Returns a pd.DataFrame with columns 'simulation', 'run', 'num_collisions', 'duration'
    """
    collisions = df_col.groupby(['simulation', 'run']).count().reset_index()
    collisions = collisions.rename({'object': 'num_collisions'}, axis=1).drop('t', axis=1)

    duration = df_ball.groupby(['simulation', 'run']).max().reset_index()[['simulation', 'run', 't']]
    duration = duration.rename({'t': 'duration'}, axis=1)
    sim_data = duration.merge(collisions)
    return sim_data


def resize(simulations, environments, max_dim=1, max_angle=1):
    # map x, y between [0, max_dim]
    # map angle between [0, max_angle
    box_width = 700
    box_height = 700
    width_coeff = max_dim / box_width
    height_coeff = max_dim / box_height
    angle_coeff = max_angle / (2 * math.pi)

    environments.triangle_x *= width_coeff
    environments.rectangle_x *= width_coeff
    environments.pentagon_x *= width_coeff

    environments.triangle_y *= height_coeff
    environments.rectangle_y *= height_coeff
    environments.pentagon_y *= height_coeff

    environments.rectangle_r *= angle_coeff
    environments.triangle_r *= angle_coeff
    environments.pentagon_r *= angle_coeff

    simulations.px *= width_coeff
    simulations.py *= height_coeff
    simulations.vx *= width_coeff
    simulations.vy *= height_coeff

    # adds polar coodinate columns
    simulations['vr'] = np.sqrt(simulations.vx ** 2 + simulations.vy ** 2)
    simulations['va'] = np.degrees(np.arctan2(simulations.vy, simulations.vx)) + 180
    simulations.va *= max_angle / 360
    return simulations, environments


def create_task_df(selected_runs, df_ball, df_env,append_t0 = True):
    """
    selected_runs: pd.DataFrame with columns 'simulation' and 'run'
    """
    selected_runs = selected_runs[['simulation', 'run']]
    simulations = selected_runs.merge(df_ball, how='inner')
    environments = df_env[df_env.simulation.isin(set(selected_runs.simulation.unique()))]
    simulations, environments = resize(simulations, environments, 10, 10) # why resize??

    if append_t0:
        environments = environments.merge(simulations[simulations.t == 0][['simulation', 'run', 'vx', 'vy']]) # add 4 columns, initial position of the ball
    else:
        environments = environments.merge(simulations[simulations.t == 0][['simulation', 'run']])

    return simulations, environments

def to_tensors(simulations, environments, device, include_v=False, outdf = False):
    """
    :param simulations:
    :param environments:
    :param device:
    :param include_v: if True, includes velocity in state, else only position
    :param outdf: output the dataframe (with dropped columns)
    :return:
    """
    sim_df = simulations.sort_values(['simulation', 'run', 't'])
    env_df = environments.sort_values(['simulation', 'run'])
    env_df = env_df.drop(['simulation', 'run', 'hole_dropped_into'], axis=1)

    # environment tensor
    envs = torch.tensor(env_df.values,
                        dtype=torch.float,
                        device=device)

    # state tensor
    states = np.zeros(((envs).shape[0], sim_df.t.max() + 1, 4 if include_v else 2))
    sim = None
    t = None
    state = None
    sim_i = 0
    for record in sim_df.to_records():
        if sim is not None and (sim, run) != (record.simulation, record.run):
            states[sim_i, t + 1:] = state
            sim_i += 1
        sim = record.simulation
        run = record.run
        t = record.t
        if include_v:
            state = [record.px, record.py, record.vx, record.vy]
        else:
            state = [record.px, record.py]
        states[sim_i, t] = state

    # if a simulation ended before max_t, pad it with the current state
    states[sim_i, t + 1:] = state

    if outdf:
        return torch.tensor(states, dtype=torch.float, device=device), envs, sim_df, env_df
    else:
        return torch.tensor(states, dtype=torch.float, device=device), envs

def create_simdata_from_samples(simulation,environment,sim_df, env_df):
    """
    :param simulation: (sim#,t,2); x and y position at each time in the simulation
    :param environment: (sim#,#env_vars), environment variables
    :return: sim_ball, sim_env structures
    """
    N_sim = simulation.shape[0]
    N_samples = simulation.shape[1]

    df_env = pd.DataFrame(environment.data.cpu().numpy(),columns = env_df.columns)
    df_ball = pd.DataFrame()

    ball_rows = []

    for i in range(N_sim):
        sim_name = 'sim_{}'.format(i)
        for smp in range(N_samples):
            row = {'simulation': sim_name, 'run': 0, 't': smp,
                   'px': simulation[i,smp,0].data.cpu().numpy(), 'py': simulation[i,smp,1].data.cpu().numpy()}
            ball_rows.append(row)

    df_ball = pd.DataFrame(ball_rows)

    return df_env, df_ball
