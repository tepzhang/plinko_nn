# Copied from https://github.com/cicl-stanford/plinko_tracking/blob/master/code/python/model/config.py


# import libraries
from __future__ import division
from math import pi
import sys
from random import choice, randint, shuffle, uniform, sample


def main():
    pass


def get_config():
    """
    Creates the main configuration file.
    Random choices: Obstacle positions and in what hole the ball is dropped
    """
    c = {}

    # PARAMETERS
    c['drop_noise'] = 0
    c['collision_noise_mean'] = 1
    c['collision_noise_sd'] = 0

    # GLOBAL SETTINGS
    c['dt'] = 1 / 60  # time step in physics engine
    c['substeps_per_frame'] = 4
    c['med'] = 350  # midpoint
    c['gravity'] = -1500  # gravity
    c['screen_size'] = {'width': 700, 'height': 700}

    # PLINKO BOX SETTINGS
    c['width'] = 600
    c['height'] = 500
    c['hole_width'] = 100
    c['hole_positions'] = [c['med'] - c['width'] / 3.5, c['med'], c['med'] + c['width'] / 3.5]
    c['wall_elasticity'] = 0.5
    c['wall_friction'] = 0.9
    c['ground_elasticity'] = 0
    c['ground_friction'] = 2
    c['ground_y'] = c['med'] - c['height'] / 2 - 5

    # BALL SETTINGS
    c['ball_radius'] = 25
    c['ball_mass'] = 10
    c['ball_elasticity'] = 0.9
    c['ball_friction'] = 0.9
    c['ball_initial_angle'] = 0

    # OBSTACLE SETTINGS
    c['obstacles'] = {
        'triangle': {
            'position': {},
            'rotation': 0,
            'size': 50,
            'n_sides': 3,
            'material': 'wood',
            'elasticity': 0.0,
            'friction': 0.0
        },
        'rectangle': {
            'position': {},
            'rotation': 0,
            'size': 50,
            'n_sides': 4,
            'material': 'wood',
            'elasticity': 0.0,
            'friction': 0.0
        },
        'pentagon': {
            'position': {},
            'rotation': 0,
            'size': 50,
            'n_sides': 5,
            'material': 'wood',
            'elasticity': 0.0,
            'friction': 0.0
        }
    }

    # random choices
    c = random_choices(c)

    return c


def random_choices(c):
    # SHAPE POSITIONS

    # 9 possible positions for the different shapes
    shape_positions = [
        {'x': c['hole_positions'][0], 'y': c['med'] + c['height'] / 4},
        {'x': c['hole_positions'][1], 'y': c['med'] + c['height'] / 4},
        {'x': c['hole_positions'][2], 'y': c['med'] + c['height'] / 4},
        {'x': c['hole_positions'][0], 'y': c['med']},
        {'x': c['hole_positions'][1], 'y': c['med']},
        {'x': c['hole_positions'][2], 'y': c['med']},
        {'x': c['hole_positions'][0], 'y': c['med'] - c['height'] / 4},
        {'x': c['hole_positions'][1], 'y': c['med'] - c['height'] / 4},
        {'x': c['hole_positions'][2], 'y': c['med'] - c['height'] / 4}
    ]

    # selected_positions = []
    # for i in range(0,3):
    # 	selected_positions.append(shape_positions[(y[i]-1)*3 + x[i] - 1])
    # shuffle(selected_positions)

    # # perturb obstacle positions and rotate
    # for idx, key in enumerate(c['obstacles']):
    # 	c['obstacles'][key]['position'] = {k : selected_positions[idx][k]  + randint(-10, 10) for k in selected_positions[idx]}
    # 	c['obstacles'][key]['rotation'] = uniform(0, 2 * pi)

    shuffle(shape_positions)  # shuffle the positions

    # perturb obstacle positions and rotate
    for idx, key in enumerate(c['obstacles']):
        c['obstacles'][key]['position'] = {k: shape_positions[idx][k] + randint(-10, 10) for k in shape_positions[idx]}
        c['obstacles'][key]['rotation'] = uniform(0, 2 * pi)

    # DROP BALL IN ONE OF THE HOLES
    c['hole_dropped_into'] = choice(range(len(c['hole_positions'])))

    return c


if __name__ == '__main__':
    main()