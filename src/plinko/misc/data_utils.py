import math
import numpy as np
import torch
import pandas as pd
from skimage import io
import os
import glob
import json
from tqdm.auto import tqdm
import random
import shutil
from . import datamaker
from . import utils
from ..simulation import utils as sim_utils


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


def make_shape_df(df_env):
    """
    Generates a dataframe that converts shape info in df_env into vertices for each shape
    Assumes that angles are in [0, 2pi]
    """
    triangles = get_env_vertices('triangle', 3, df_env)
    rectangles = get_env_vertices('rectangle', 4, df_env)
    pentagons = get_env_vertices('pentagon', 5, df_env)

    shapes = df_env[['simulation']]
    add_shapes(shapes, 'triangle', triangles)
    add_shapes(shapes, 'rectangle', rectangles)
    add_shapes(shapes, 'pentagon', pentagons)
    return shapes


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


def create_dataframes(image_sim_path, save_path, valid_f=.1, test_f=.1, max_duration=300):
    """
    Creates df_env, df_ball, and sim_data
    :param image_sim_path: path of image_sim
    :param save_path: path to save the dataframes
    :param max_duration: filters out simulations with longer duration than this
    :return:
    """
    config_raw = {}
    sim_raw = {}
    rows = []

    print("Reading raw files")
    for path in tqdm(sorted(glob.glob(os.path.join(image_sim_path, '**/**/')))):
        simulation, run = path.split('/')[-2].split('_')
        simulation = int(simulation[1:])
        run = int(run[1:])

        with open(os.path.join(path, 'config.json')) as f:
            config_raw[simulation] = json.load(f)
        with open(os.path.join(path, 'sim.json')) as f:
            sim_raw[(simulation, run)] = json.load(f)
        for img in sorted(glob.glob(path + '/*.bmp')):
            t = int(img.split('/')[-1].split('.')[0])
            rows.append({'simulation': simulation,
                         'run': run,
                         't': t,
                         'imgpath': os.path.join(*img.split('/')[-3:])})

    simulations = pd.DataFrame(rows).sort_values(['simulation', 'run', 't'])

    print("Loading env, ball, and col data")
    # load config and env data into dataframes
    df_env, df_ball, df_col = datamaker.get_sim_df(config_raw, sim_raw)
    df_ball = df_ball.merge(simulations)
    sim_data = get_sim_data(df_ball, df_col)

    # filter by max duration
    selected_runs = sim_data[(sim_data.duration <= max_duration)]
    df_ball = df_ball.merge(selected_runs[['simulation', 'run']])
    df_env = df_env.merge(selected_runs[['simulation']].drop_duplicates())

    # appends the collision column
    print("Appending collision data")
    collisions = {}
    for s, r, o, t in df_col.values:
        collisions[(s, r, t)] = o
    column = []
    for record in tqdm(df_ball.to_records()):
        key = (record.simulation, record.run, record.t)
        if key not in collisions:
            column.append('none')
        else:
            c = collisions[key]
            if c == 'walls':
                c = 'left_wall' if record.px < .1 else 'right_wall'
            column.append(c)
    df_ball['collision'] = column

    # pads the dataframe with last step for each simulation until all simulations have same max step
    print("Padding timesteps")
    indices = df_ball.groupby(['simulation', 'run'], sort=False)['t'].transform('max') == df_ball.t
    last_t = df_ball[indices]
    max_t = last_t.t.max()
    pads = []

    for record in tqdm(last_t.to_records(index=False)):
        for t in range(max_t - record.t):
            r = record.copy()
            r.t += t + 1
            pads.append(r)
    pads = pd.DataFrame(np.array(pads))
    df_ball = df_ball.append(pads).sort_values(['simulation', 'run', 't'])

    # split train, valid, and test sets
    test_boundary = int((1 - test_f) * len(sim_data.simulation.unique()))
    valid_boundary = int((1 - test_f - valid_f) * len(sim_data.simulation.unique()))
    df_ball['dataset'] = [('train' if s < valid_boundary else 'valid') if s < test_boundary else 'test'
                       for s in df_ball.simulation]
    df_env['dataset'] = [('train' if s < valid_boundary else 'valid') if s < test_boundary else 'test'
                       for s in df_env.simulation]
    sim_data['dataset'] = [('train' if s < valid_boundary else 'valid') if s < test_boundary else 'test'
                       for s in sim_data.simulation]

    # save
    df_ball = df_ball.sort_values(['simulation', 'run', 't']).reset_index(drop=True)
    df_env = df_env.sort_values(['simulation']).reset_index(drop=True)
    sim_data.to_feather(os.path.join(save_path, 'sim_data.feather'))
    df_ball.to_feather(os.path.join(save_path, 'df_ball.feather'))
    df_env.to_feather(os.path.join(save_path, 'df_env.feather'))
    return df_env, df_ball, sim_data


def select_runs(df_env, df_ball, sim_data):
    """
    Filters out runs from df_ball and df_env not in sim_data
    usage:
    >>> selected = sim_data[sim_data.num_collisions == 3]
    >>> df_env, df_ball = select_runs(df_env, df_ball, sim_data)
    """
    df_ball = df_ball.merge(sim_data[['simulation', 'run']]).sort_values(['simulation', 'run', 't'])
    df_env = df_env.merge(sim_data[['simulation']].drop_duplicates()).sort_values('simulation')

    max_duration = df_ball[(df_ball.vx == 0) & (df_ball.vy == 0)].groupby(['simulation', 'run'],
                                                                          as_index=False).min().t.max()
    df_ball = df_ball[df_ball.t <= max_duration]
    return df_env, df_ball


def sample_batch(df_env, df_ball, dataset: str, batch_size: int, no_images=False, device='cpu'):
    """
    Samples a random batch and returns tensors
    :param df_env:
    :param df_ball:
    :param dataset:
    :param batch_size:
    :param no_images:
    :param device:
    :return:
    """
    assert dataset in ('train', 'valid', 'test')
    batch_keys = df_ball[df_ball.dataset == dataset][['simulation', 'run']].sample(batch_size)
    batch_env = df_env.merge(batch_keys)
    batch_balls = df_ball.merge(batch_keys)
    batch_shapes = [batch_env[['triangle_x', 'triangle_y', 'triangle_r']],
                    batch_env[['rectangle_x', 'rectangle_y', 'rectangle_r']],
                    batch_env[['pentagon_x', 'pentagon_y', 'pentagon_r']]]
    shape_one_hot = utils.expand_along_dim(torch.eye(3, device=device), batch_size, 1)
    for i in range(3):
        tensor = torch.tensor(batch_shapes[i].values, dtype=torch.float, device=device)
        batch_shapes[i] = torch.cat([tensor, shape_one_hot[i]], dim=-1)
    triangles, rectangles, pentagons = batch_shapes

    max_t = df_ball.t.max()
    balls = batch_balls[['px', 'py', 'vx', 'vy']].values
    balls = torch.tensor(balls, dtype=torch.float, device=device).view((-1, max_t + 1, 4))

    if no_images:
        images = None
    else:
        images = load_images(batch_balls.imgpath, device)
        images = images.view(-1, max_t + 1, 100, 100)

    return triangles, rectangles, pentagons, balls, images


def save_batches(df_ball, df_env, dataset, batch_size, save_path, device):
    """
    Saves batch tensors for fast loading later
    :param df_ball:
    :param df_env:
    :param dataset:
    :param batch_size:
    :param save_path:
    :param device:
    :return:
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)

    df_ball = df_ball[df_ball.dataset == dataset]
    max_t = df_ball.t.max() + 1
    for run in tqdm(df_ball.run.unique()):

        run_ball = df_ball[(df_ball.run == run)]
        run_simulations = set(run_ball.simulation)
        run_env = df_env[df_env.simulation.isin(run_simulations)]

        num_simulations = len(run_simulations)

        balls = torch.tensor(run_ball[['px', 'py', 'vx', 'vy']].values, dtype=torch.float, device=device)
        balls = balls.view(num_simulations, max_t, -1)
        triangles, rectangles, pentagons = make_shape_tensors(run_env, device)
        run_imgpaths = run_ball.imgpath.values.reshape(num_simulations, -1)

        srun = str(run).zfill(2)
        for i in tqdm(range(0, num_simulations, batch_size)):
            j = i + batch_size
            batch_balls = balls[i:j]
            batch_triangles = triangles[i:j]
            batch_rectangles = rectangles[i:j]
            batch_pentagons = pentagons[i:j]
            batch_imgpaths = run_imgpaths[i:j]
            images = load_images(batch_imgpaths.flatten(), device=device)
            images = images.view(len(batch_imgpaths), max_t, 100, 100)

            sbatch = str(i).zfill(3)
            torch.save(batch_balls, os.path.join(save_path, 'balls_{}_{}.pt'.format(srun, sbatch)))
            torch.save(batch_triangles, os.path.join(save_path, 'triangles_{}_{}.pt'.format(srun, sbatch)))
            torch.save(batch_rectangles, os.path.join(save_path, 'rectangles_{}_{}.pt'.format(srun, sbatch)))
            torch.save(batch_pentagons, os.path.join(save_path, 'pentagons_{}_{}.pt'.format(srun, sbatch)))
            torch.save(images, os.path.join(save_path, 'images_{}_{}.pt'.format(srun, sbatch)))


def load_batch(save_path, run, batch, load_images=True, device='cpu'):
    """
    Load saved batches from save_batch
    :param batch_path:
    :param run:
    :param batch:
    :param device:
    :return:
    """
    run = str(run).zfill(2)
    batch = str(batch).zfill(3)
    device = 'cuda:{}'.format(device)
    loader = lambda s: torch.load(os.path.join(save_path, '{}_{}_{}.pt'.format(s, run, batch)), map_location=device)

    files = ['triangles', 'rectangles', 'pentagons', 'balls']
    triangles, rectangles, pentagons, balls = [loader(s) for s in files]
    images = loader('images') if load_images else None
    return triangles, rectangles, pentagons, balls, images


def get_batch_keys(save_path):
    batch_keys = sorted(set([tuple(s[:-3].split('_')[1:]) for s in os.listdir(save_path)]))
    return batch_keys # list of (run, batch)


################ Helper Functions ###############

def make_shape_tensors(df_env, device):
    """
    Takes a df_env and returns tensors for each shape
    Used by save_batches
    :param df_env:
    :return:
    """
    batch_shapes = [df_env[['triangle_x', 'triangle_y', 'triangle_r']],
                    df_env[['rectangle_x', 'rectangle_y', 'rectangle_r']],
                    df_env[['pentagon_x', 'pentagon_y', 'pentagon_r']]]
    shape_one_hot = utils.expand_along_dim(torch.eye(3, device=device), len(df_env), 1)
    for i in range(3):
        tensor = torch.tensor(batch_shapes[i].values, dtype=torch.float, device=device)
        batch_shapes[i] = torch.cat([tensor, shape_one_hot[i]], dim=-1)
    triangles, rectangles, pentagons = batch_shapes
    return triangles, rectangles, pentagons


def add_shapes(shapes_df, shape_name, vertices):
    """
    Helper function for make_shape_df
    vertices shape: [batch_size, ngon, 2]
    """
    for i in range(vertices.shape[1]):
        shapes_df['{}_vx{}'.format(shape_name, i)] = vertices[:, i, 0]
        shapes_df['{}_vy{}'.format(shape_name, i)] = vertices[:, i, 1]


def get_env_vertices(shape, ngon, df_env):
    """
    Helper function for make_shape_df
    """
    vertices = []
    for x, y, r in df_env[[shape + '_x', shape + '_y', shape + '_r']].values:
        vertices.append(sim_utils.get_vertices(ngon, 50, x, y, r))
    vertices = np.stack(vertices)
    return vertices


def load_images(imgpaths, device='cpu', show_tqdm=False):
    """
    :param imgpaths: list of paths to image files
        (currently intended for bmp files, may produce incorrect results if not bitmaps)
    :param device: a tensor with the images loaded
    :return:
    """
    imgs = []
    iterator = tqdm(imgpaths) if show_tqdm else imgpaths
    for imgpath in iterator:
        imgs.append(io.imread(imgpath, as_gray=True))
    return 1 - torch.tensor(np.stack(imgs, axis=0), dtype=torch.float, device=device)

