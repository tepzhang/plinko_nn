# Copied from https://github.com/cicl-stanford/plinko_tracking/blob/master/code/python/model/engine.py


# import libraries
from __future__ import division
import numpy as np
import pymunk
from pymunk import Vec2d

# import files
from . import utils
# from . import visual

shape_code = {'walls': 0,
              'ground': 1,
              'ball': 2,
              'rectangle': 3,
              'triangle': 4,
              'pentagon': 5}

inverse_shape_code = {shape_code[key]: key for key in shape_code}


def main():
    # c = config.get_config() # generate config
    c = utils.load_config("data/cases/ground_truth/world_309.json")  # load a config file
    # c = misc.load_config("data/json/world_6541.json")  # load a config file
    c['hole_dropped_into'] = 1
    c['drop_noise'] = 0
    c['collision_noise_mean'] = 1
    c['collision_noise_sd'] = 0
    # for i in range(0,5):
    # print(c)
    data = run_simulation(c)
    visual.visualize(c, data)


# tmp = misc.loss(prop = 0, target = 50, sd = 2)
# print("tmp", tmp)
# pass
# path_screenshot_write = 'data/images/pygame'
# for world in range(1,201):
# world = 6
# for i in range(0,5):
# 	c = misc.load_config("data/json/world_" + str(world) + ".json")
# 	c['hole_dropped_into'] = 1
# 	# c['drop_noise'] = 0
# 	# c['collision_noise_mean'] = 1
# 	# c['collision_noise_sd'] = 0
# 	data = run_simulation(c)
# 	visual.visualize(c, data)
# 	# visual.snapshot(c, image_path = path_screenshot_write + "/", image_name = 'world' + str(world)) #save snapshot

def run_simulation(c):
    # PHYSICS PARAMETERS
    space = pymunk.Space()
    space.gravity = (0.0, c['gravity'])

    # noise applied to how the ball is dropped
    ball_drop_noise(c, sd=c['drop_noise'])
    # print("sd = c['drop_noise']", c['drop_noise'])

    # CREATE OBJECTS
    make_walls(c, space)
    make_obstacles(c, space)
    ball = make_ball(c, space)
    make_ground(c, space)

    # CREATE COLLISION HANDLERS
    h = space.add_wildcard_collision_handler(shape_code['ball'])
    # h.post_solve = record_collision #records each time step at which collision occurs
    h.begin = record_collision  # records only the beginning of each collision
    h.data['collisions'] = []

    # set ball to velocity 0 when it hits the ground
    g = space.add_collision_handler(shape_code['ground'], shape_code['ball'])
    g.post_solve = record_collision
    g.post_solve = ground_collision
    g.data['running'] = True
    g.data['ball'] = ball

    # jitter velocity at end of every collision
    for ob in ['rectangle', 'triangle', 'pentagon']:
        ch = space.add_collision_handler(shape_code['ball'], shape_code[ob])
        ch.separate = jitter_velocity
        ch.data['ball'] = ball
        ch.data['collision_noise_sd'] = c['collision_noise_sd']
        ch.data['collision_noise_mean'] = c['collision_noise_mean']

    ###############
    ## MAIN LOOP ##
    ###############
    timestep = 0

    all_data = {}

    ball_pos = []
    ball_vel = []

    while g.data['running']:  # run into ground collision callback
        ### Update physics

        for _ in range(c['substeps_per_frame']):
            space.step(c['dt'] / c['substeps_per_frame'])

        # space.step(c['dt']) #original step function
        timestep += 1
        # space.step(c['dt'])
        ball_pos.append({'x': ball.position.x,
                         'y': ball.position.y})
        ball_vel.append({'x': ball.velocity.x,
                         'y': ball.velocity.y})
        h.data['current_timestep'] = timestep

    # clean up collisions
    collisions = clean_collisions(collisions=h.data['collisions'])
    all_data['collisions'] = collisions
    all_data['ball_position'] = ball_pos
    all_data['ball_velocity'] = ball_vel

    return all_data


def make_ball(c, space):
    inertia = pymunk.moment_for_circle(c['ball_mass'], 0, c['ball_radius'], (0, 0))
    body = pymunk.Body(c['ball_mass'], inertia)
    x = c['hole_positions'][c['hole_dropped_into']]
    y = c['med'] + c['height'] / 2 + c['ball_radius']
    body.position = x, y
    shape = pymunk.Circle(body, c['ball_radius'], (0, 0))
    shape.elasticity = c['ball_elasticity']
    shape.friction = c['ball_friction']

    shape.collision_type = shape_code['ball']

    space.add(body, shape)

    # used for setting initial velocity (should not be part of the ball definition)
    ang = c['ball_initial_angle']
    amp = 100  #
    off = 3 * np.pi / 2
    # so that clockwise is negative
    body.velocity = Vec2d(amp * -np.cos(ang + off), amp * np.sin(ang + off))

    return body


def make_ground(c, space):
    sz = (c['width'], 10)  # 4is for border

    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = (c['med'], c['ground_y'])

    shape = pymunk.Poly.create_box(body, sz)
    shape.elasticity = 1
    shape.friction = 1

    shape.collision_type = shape_code['ground']
    space.add(body, shape)


def make_walls(c, space):
    walls = pymunk.Body(body_type=pymunk.Body.STATIC)

    topwall_y = c['med'] + c['height'] / 2

    static_lines = [
        # top horizontal: 1
        pymunk.Segment(walls,
                       a=(c['med'] - c['width'] / 2, topwall_y),
                       b=(c['hole_positions'][0] - c['hole_width'] / 2, topwall_y),
                       radius=2.0),
        # top horizontal: 2
        pymunk.Segment(walls,
                       a=(c['hole_positions'][0] + c['hole_width'] / 2, topwall_y),
                       b=(c['hole_positions'][1] - c['hole_width'] / 2, topwall_y),
                       radius=2.0),
        # top horizontal: 3
        pymunk.Segment(walls,
                       a=(c['hole_positions'][1] + c['hole_width'] / 2, topwall_y),
                       b=(c['hole_positions'][2] - c['hole_width'] / 2, topwall_y),
                       radius=2.0),
        # top horizontal: 4
        pymunk.Segment(walls,
                       a=(c['hole_positions'][2] + c['hole_width'] / 2, topwall_y),
                       b=(c['med'] + c['width'] / 2, topwall_y),
                       radius=2.0),

        # left vertical
        pymunk.Segment(walls,
                       a=(c['med'] - c['width'] / 2, c['med'] - c['height'] / 2),
                       b=(c['med'] - c['width'] / 2, c['med'] + c['height'] / 2),
                       radius=2.0),

        # right vertical
        pymunk.Segment(walls,
                       a=(c['med'] + c['width'] / 2, c['med'] - c['height'] / 2),
                       b=(c['med'] + c['width'] / 2, c['med'] + c['height'] / 2),
                       radius=2.0)]

    for line in static_lines:
        line.elasticity = c['wall_elasticity']
        line.friction = c['wall_friction']
        line.collision_type = shape_code['walls']

    space.add(walls, static_lines)


def make_obstacles(c, space):
    for ob in c['obstacles']:
        rigid_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        polygon = utils.generate_ngon(c['obstacles'][ob]['n_sides'], c['obstacles'][ob]['size'])
        shape = pymunk.Poly(rigid_body, polygon)
        shape.elasticity = c['obstacles'][ob]['elasticity']
        shape.friction = c['obstacles'][ob]['friction']

        pos = c['obstacles'][ob]['position']
        rigid_body.position = pos['x'], pos['y']
        rigid_body.angle = c['obstacles'][ob]['rotation']

        shape.collision_type = shape_code[ob]  # key ob is the name

        space.add(shape, rigid_body)


### CALLBACKS

# records collisions between the ball and obstacles/walls
def record_collision(arbiter, space, data):
    ob1 = inverse_shape_code[int(arbiter.shapes[0].collision_type)]
    ob2 = inverse_shape_code[int(arbiter.shapes[1].collision_type)]
    # data['collisions'].append((data['current_timestep'], ob1, ob2))
    data['collisions'].append({'objects': (ob1, ob2), 'step': data['current_timestep']})
    return True


# records when the ball hits the ground
def ground_collision(arbiter, space, data):
    data['ball'].velocity = Vec2d(0, 0)
    data['running'] = False
    return True


def jitter_velocity(arbiter, space, data):
    mult = utils.gaussian_noise(data['collision_noise_mean'],
                                data['collision_noise_sd'])  # potentially asymmetric noise

    # change magnitude of velocity
    cur_vel = data['ball'].velocity
    new_vel = Vec2d(cur_vel.x * mult, cur_vel.y * mult)
    data['ball'].velocity = new_vel


def clean_collisions(collisions):
    """
    Interactions with shapes sometimes result in multiple collisions, but we only want to keep the first collision.
    """
    idx = 1
    while idx < len(collisions):
        if collisions[idx - 1]['objects'] == collisions[idx]['objects']:
            del collisions[idx]
        else:
            idx += 1

    return (collisions)


def ball_drop_noise(c, sd):
    """ Noise applied to the angle in which the ball is dropped. """
    c['ball_initial_angle'] = utils.gaussian_noise(0, sd)


def ball_collision_noise(c, mean, sd):
    """
    Noise applied to the ball's velocity after collisions.
    A value of 1 means no noise.
    """
    c['collision_noise_mean'] = mean
    c['collision_noise_sd'] = sd


if __name__ == '__main__':
    main()
