# Copied from https://github.com/cicl-stanford/plinko_tracking/blob/master/code/python/model/visual.py


# import libraries
from __future__ import division
import numpy as np
import json

# import files
from . import config


def main():
    # combine_video_and_audio(video = 'data/videos/test.mp4', audio = 'data/wav/test.wav')
    tmp = loss(prop=60, target=63, sd=2)
    print("tmp", tmp)


def generate_ngon(n, rad):
    """ Function to generate ngons (e.g. Pentagon) """
    pts = []
    ang = 2 * np.pi / n
    for i in range(n):
        pts.append((np.sin(ang * i) * rad, np.cos(ang * i) * rad))
    return pts


def gaussian_noise(mean=0, sd=1):
    """ Apply gaussian noise """
    return np.random.normal(mean, sd)


def loss(prop, target, sd=100):
    """ Gaussian loss """
    return -((prop - target) / sd) ** 2


def flipy(c, y):
    """Small hack to convert chipmunk physics to pygame coordinates"""
    return -y + c['screen_size']['height']


def load_config(name):
    c = config.get_config()

    with open(name, 'rb') as f:
        ob = json.load(f)

    # PARAMETERS
    c['drop_noise'] = ob['parameters']['drop_noise']
    c['collision_noise_mean'] = ob['parameters']['collision_noise_mean']
    c['collision_noise_sd'] = ob['parameters']['collision_noise_sd']
    # c['loss_sd_vision'] = ob['parameters']['loss_sd_vision']
    # c['loss_sd_sound'] = ob['parameters']['loss_sd_sound']
    # c['loss_penalty_sound'] = ob['parameters']['loss_penalty_sound']

    # GLOBAL SETTINGS
    c['dt'] = ob['global']['timestep']
    c['substeps_per_frame'] = ob['global']['substeps']
    c['med'] = ob['global']['midpoint']
    c['gravity'] = ob['global']['gravity']
    c['screen_size'] = ob['global']['screen_size']
    c['hole_dropped_into'] = ob['global']['hole_dropped_into'] - 1
    # c['hole_dropped_into'] = ob['global']['hole_dropped_into']

    # PLINKO BOX SETTINGS
    c['width'] = ob['box']['width']
    c['height'] = ob['box']['height']
    c['hole_width'] = ob['box']['holes']['width']
    c['hole_positions'] = ob['box']['holes']['positions']
    c['wall_elasticity'] = ob['box']['walls']['elasticity']
    c['wall_friction'] = ob['box']['walls']['friction']
    c['ground_elasticity'] = ob['box']['ground']['elasticity']
    c['ground_friction'] = ob['box']['ground']['friction']
    c['ground_y'] = ob['box']['ground']['position']['y']

    # BALL SETTINGS
    c['ball_radius'] = ob['ball']['radius']
    c['ball_mass'] = ob['ball']['mass']
    c['ball_elasticity'] = ob['ball']['elasticity']
    c['ball_friction'] = ob['ball']['friction']

    # OBSTACLE SETTINGS
    c['obstacles'] = ob['obstacles']

    return c


def combine_video_and_audio(video, audio, filename='test_with_sound.mp4'):
    """
    Function to combine video and audio.
    video = path to mp4 file
    audio = path to wav file
    filename = path to created file
    """
    import subprocess as sp
    sp.call('ffmpeg -i {video} -i {audio} -c:v copy -c:a aac -strict experimental {filename}'.format(video=video,
                                                                                                     audio=audio,
                                                                                                     filename=filename),
            shell=True)


if __name__ == '__main__':
    main()