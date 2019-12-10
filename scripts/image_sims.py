repo_path = '~/Desktop/plinko_nn'

import sys
sys.path.append(repo_path + '/src')
import warnings
warnings.filterwarnings('ignore')

from plinko.simulation.config import get_config
from plinko.simulation.visual import visualize
from plinko.simulation.engine import run_simulation
import json
from tqdm.auto import tqdm

for i in tqdm(range(1000)):
    config = get_config(100, wall_buffer=False)
    for j in range(20):
        sim = run_simulation(config)
        save_path = '../data/simulations/image_sim/w{}_r{}'.format(i, j)
        visualize(config, sim, image_path=save_path)
        with open(save_path + '/config.json', 'w') as f:
            json.dump(config, f)
        with open(save_path + '/sim.json', 'w') as f:
            json.dump(sim, f)
