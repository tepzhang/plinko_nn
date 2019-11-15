# Copied from https://github.com/cicl-stanford/plinko_tracking/blob/master/code/python/model/visual.py



# import libraries
import os
import shutil
import pygame
from numpy import array, cos, dot, pi, sin
from pygame.color import THECOLORS
from pygame.constants import QUIT, KEYDOWN, K_ESCAPE

# import files
from . import config
from . import engine
from . import utils


def main():
    c = config.get_config()
    # c = misc.load_config("data/json/world_1777.json")
    c = utils.load_config("data/json/world_6.json")
    sim_data = engine.run_simulation(c)
    visualize(c, sim_data)


def visualize(c, sim_data, save_images=False, make_video=False, video_name='test'):
    """
    Visualize the world.
    save_images: set True to save images
    make_video: set True to make a video
    """

    # setup pygame
    screen = pygame.display.set_mode((c['screen_size']['width'], c['screen_size']['height']))

    # set up the rotated obstacles
    rotated = rotate_shapes(c)

    # make temporary directory for images
    if save_images:
        img_dir = 'images_temp'
        try:
            shutil.rmtree(img_dir)
        except:
            pass
        os.mkdir(img_dir)

    for t, frame in enumerate(sim_data['ball_position']):
        screen.fill(THECOLORS['white'])

        colors = [THECOLORS['blue'], THECOLORS['red'], THECOLORS['green']]

        # draw objects
        draw_obstacles(rotated, screen, colors)
        draw_ground(c, screen)
        draw_ball(c, screen, frame)
        draw_walls(c, screen)

        pygame.event.get()  # this is just to get pygame to update
        pygame.display.flip()
        # pygame.time.wait(1)

        if save_images:
            pygame.image.save(screen, os.path.join(img_dir, '%05d' % t + '.png'))

    if make_video:
        import subprocess as sp
        sp.call(
            'ffmpeg -y -framerate 60 -i {ims}  -c:v libx264 -profile:v high -crf 10 -pix_fmt yuv420p {videoname}.mp4'.format(
                ims=img_dir + "/\'%05d.png\'", videoname="data/videos/" + video_name), shell=True)
        shutil.rmtree(img_dir)  # remove temporary directory

    running = True

    while running:
        for e in pygame.event.get():
            if e.type == QUIT:
                running = False
            elif e.type == KEYDOWN and e.key == K_ESCAPE:
                running = False


def snapshot(c, image_path, image_name):
    """
    Create a snapshot of the world with the obstacles
    """
    # setup pygame
    # pygame.init()

    screen = pygame.Surface((c['screen_size']['width'], c['screen_size']['height']))
    screen.fill(THECOLORS['white'])

    colors = [THECOLORS['blue'], THECOLORS['red'], THECOLORS['green']]

    # set up the rotated obstacles
    rotated = rotate_shapes(c)

    # draw objects
    draw_obstacles(rotated, screen, colors)
    draw_ground(c, screen)
    draw_walls(c, screen)

    # save image
    pygame.image.save(screen, os.path.join(image_path, image_name + '.png'))


# quit pygame
# pygame.quit()

##############
# HELPER FUNCTIONS
##############

def rotate_shapes(c):
    # set up rotated shapes
    rotated = {name: [] for name in c['obstacles']}
    for shape in c['obstacles']:

        poly = utils.generate_ngon(c['obstacles'][shape]['n_sides'],
                                   c['obstacles'][shape]['size'])

        ob_center = array([c['obstacles'][shape]['position']['x'],
                           utils.flipy(c, c['obstacles'][shape]['position']['y'])])

        rot = c['obstacles'][shape]['rotation'] + pi
        for p in poly:
            rotmat = array([[cos(rot), sin(rot)],
                            [-sin(rot), cos(rot)]])

            rotp = dot(rotmat, p)

            rotp += ob_center

            rotated[shape].append(rotp)
    return rotated


def draw_ball(c, screen, frame):
    pygame.draw.circle(screen,
                       THECOLORS['black'],
                       (int(frame['x']), int(utils.flipy(c, frame['y']))),
                       c['ball_radius'])


def draw_walls(c, screen):
    # top horizontal: 1
    topwall_y = utils.flipy(c, c['med'] + c['height'] / 2)

    pygame.draw.line(screen,
                     THECOLORS['black'],
                     (c['med'] - c['width'] / 2, topwall_y),
                     (c['hole_positions'][0] - c['hole_width'] / 2, topwall_y))

    # top horizontal: 2
    pygame.draw.line(screen,
                     THECOLORS['black'],
                     (c['hole_positions'][0] + c['hole_width'] / 2, topwall_y),
                     (c['hole_positions'][1] - c['hole_width'] / 2, topwall_y))

    # top horizontal: 3
    pygame.draw.line(screen,
                     THECOLORS['black'],
                     (c['hole_positions'][1] + c['hole_width'] / 2, topwall_y),
                     (c['hole_positions'][2] - c['hole_width'] / 2, topwall_y))

    # top horizontal: 4
    pygame.draw.line(screen,
                     THECOLORS['black'],
                     (c['hole_positions'][2] + c['hole_width'] / 2, topwall_y),
                     (c['med'] + c['width'] / 2, topwall_y))

    # left vertical
    pygame.draw.line(screen,
                     THECOLORS['black'],
                     (c['med'] - c['width'] / 2, c['med'] - c['height'] / 2),
                     (c['med'] - c['width'] / 2, c['med'] + c['height'] / 2))

    # right vertical
    pygame.draw.line(screen,
                     THECOLORS['black'],
                     (c['med'] + c['width'] / 2, c['med'] - c['height'] / 2),
                     (c['med'] + c['width'] / 2, c['med'] + c['height'] / 2))


def draw_ground(c, screen):
    pygame.draw.line(screen,
                     THECOLORS['black'],
                     (c['med'] - c['width'] / 2, c['med'] + c['height'] / 2),
                     (c['med'] + c['width'] / 2, c['med'] + c['height'] / 2))


def draw_obstacles(rotated, screen, colors):
    for idx, shape in enumerate(rotated):
        pygame.draw.polygon(screen, colors[idx], rotated[shape])


if __name__ == '__main__':
    main()
