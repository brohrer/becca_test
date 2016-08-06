"""
Connect a Becca brain to a world and run them.

Command line usage
-----
Test Becca on the grid1D.py world.
    python -m test grid1D
        or
    python -m test 1

Test Becca on the suite of all test worlds.
    python -m test all
        or
    python -m test 0

Profile Becca on the image2D.py world.
    python -m test image2D --profile
        or
    python -m test 9 -p
"""

from __future__ import print_function
import argparse
import cProfile
import pstats
import time

import matplotlib.pyplot as plt
import numpy as np

import becca.connector
# Import the suite of test worlds
from becca_test.grid_1D import World as World_grid_1D
from becca_test.grid_1D_chase import World as World_grid_1D_chase
from becca_test.grid_1D_delay import World as World_grid_1D_delay
from becca_test.grid_1D_ms import World as World_grid_1D_ms
from becca_test.grid_1D_noise import World as World_grid_1D_noise
from becca_test.grid_2D import World as World_grid_2D
from becca_test.grid_2D_dc import World as World_grid_2D_dc
from becca_test.image_1D import World as World_image_1D
from becca_test.image_2D import World as World_image_2D
from becca_test.fruit import World as World_fruit


def suite(lifespan=1e4):
    """
    Run all the worlds in the benchmark and tabulate their performance.
    """
    start_time = time.time()
    performance = []
    performance.append(test_world(World_grid_1D, lifespan=lifespan))
    performance.append(test_world(World_grid_1D_chase, lifespan=lifespan))
    performance.append(test_world(World_grid_1D_delay, lifespan=lifespan))
    performance.append(test_world(World_grid_1D_ms, lifespan=lifespan))
    performance.append(test_world(World_grid_1D_noise, lifespan=lifespan))
    performance.append(test_world(World_grid_2D, lifespan=lifespan))
    performance.append(test_world(World_grid_2D_dc, lifespan=lifespan))
    performance.append(test_world(World_image_1D, lifespan=lifespan))
    performance.append(test_world(World_image_2D, lifespan=lifespan))
    performance.append(test_world(World_fruit, lifespan=lifespan))
    finish_time = time.time()

    # Some tests are harder than others. Weight them accordingly.
    weights = np.array([
        1., # grid1D
        1., # grid1D_chase
        1., # grid1D_delay
        1., # grid1D_ms
        1., # grid1D_noise
        3., # grid2D
        4., # grid2D_dc
        5., # image_1D
        10., # image_2D
        3., # fruit
        ])

    print('Individual test world scores:')
    scores = []
    for score in performance:
        print('    {0:.2}, {1}'.format(score[0], score[1]))
        scores.append(score[0])
    mean_score = np.sum(np.array(scores) * weights) / np.sum(weights)
    print('Weighted test suite score: {0:.2}'.format(mean_score))
    print('Test suite completed in {0:.2} seconds ({1:.2} minutes)'.format(
        finish_time - start_time, (finish_time - start_time) / 60.))

    # Block the program, displaying all plots.
    # When the plot windows are closed, the program closes.
    plt.show()


def test_world(world_class, lifespan=1e4):
    """
    Test the brain's performance on a world.

    Parameters
    ----------
    world_class : World
        The class containing the Becca-compatible world that the
        brain will be receiving sensor and reward information from and
        sending action commands to.
    testing_period : int, optional
        The number of time steps to test the brain
        on the current world.

    Returns
    -------
    performance : float
        The average reward per time step during the testing period.
    world.name : str
        The name of the world that was run.
    """
    start_time = time.time()
    world = world_class(lifespan=lifespan)
    performance = becca.connector.run(world)
    finish_time = time.time()
    delta_time = finish_time - start_time
    print('Performance is: {0:.3}'.format(performance))
    print(world.name, 'ran in {0:.2} seconds ({1:.2} minutes),'.format(
        delta_time, delta_time / 60.))
    print('an average of {0:.2} seconds ({1:.2} ms) per time step.'.format(
        delta_time / lifespan, 1000. * delta_time / lifespan))
    return performance, world.name


def profile(World, lifespan=1e4):
    """
    Profile the brain's performance on the selected world.
    """
    print('Profiling Becca\'s performance...')
    command = 'becca.connector.run(World(lifespan={0}), restore=True)'.format(
        lifespan)
    cProfile.run(command, 'becca_test.profile')
    profile_stats = pstats.Stats('becca_test.profile')
    profile_stats.strip_dirs().sort_stats('time', 'cumulative').print_stats(30)
    print('   View at the command line with')
    print(' > python -m pstats becca_test.profile')


if __name__ == '__main__':
    # Build the command line parser.
    parser = argparse.ArgumentParser(
        description='Test Becca on some toy worlds.')
    parser.add_argument('world', default='all',
                        help=' '.join(['The test world to run.',
                                       'Choose by name or number:',
                                       '1) grid1D,',
                                       '2) grid1D_chase,',
                                       '3) grid1D_delay,',
                                       '4) grid1D_ms,',
                                       '5) grid1D_noise,',
                                       '6) grid2D,',
                                       '7) grid2D_dc,',
                                       '8) image1D,',
                                       '9) image2D,',
                                       '10) fruit,',
                                       '0) all',
                                       'Default value is all.']))
    parser.add_argument(
        '-p', '--profile', action='store_true',
        help="Profile Becca's performance.")
    parser.add_argument(
        '-t', '--lifespan', type=int,
        help='The number of time steps (in thousands) to run the world.')
    args = parser.parse_args()

    if args.world == 'grid1D' or args.world == '1':
        World = World_grid_1D
    elif args.world == 'grid1D_chase' or args.world == '2':
        World = World_grid_1D_chase
    elif args.world == 'grid1D_delay' or args.world == '3':
        World = World_grid_1D_delay
    elif args.world == 'grid1D_ms' or args.world == '4':
        World = World_grid_1D_ms
    elif args.world == 'grid1D_noise' or args.world == '5':
        World = World_grid_1D_noise
    elif args.world == 'grid2D' or args.world == '6':
        World = World_grid_2D
    elif args.world == 'grid2D_dc' or args.world == '7':
        World = World_grid_2D_dc
    elif args.world == 'image1D' or args.world == '8':
        World = World_image_1D
    elif args.world == 'image2D' or args.world == '9':
        World = World_image_2D
    elif args.world == 'fruit' or args.world == '10':
        World = World_fruit
    else:
        args.world = 'all'

    if args.lifespan is None:
        lifespan = 1e4
    else:
        lifespan = args.lifespan * 1000

    if args.world == 'all':
        suite(lifespan=lifespan)
    elif args.profile:
        profile(World, lifespan=lifespan)
    else:
        performance = becca.connector.run(World, restore=True)
        print('performance:', performance)
