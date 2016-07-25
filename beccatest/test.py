"""
Connect a BECCA brain to a world and run them.

Usage
-----
Test BECCA on the grid1D.py world.
    > python -m test grid1D
        or
    > python -m test 1

Test BECCA on the suite of all test worlds.
    > python -m test all
        or
    > python -m test 0

Profile BECCA on the image2D.py world.
    > python -m test image2D --profile 
        or
    > python -m test 9 -p 
"""

from __future__ import print_function
import argparse
import cProfile
import matplotlib.pyplot as plt
import pstats
import time

import numpy as np

from becca.core.brain import Brain

TEST_LENGTH = 2e4


def suite():
    """
    Run all the worlds in the benchmark and tabulate their performance.
    """
    # Import the suite of test worlds
    from beccatest.worlds.grid_1D import World as World_grid_1D
    from beccatest.worlds.grid_1D_chase import World as World_grid_1D_chase
    from beccatest.worlds.grid_1D_delay import World as World_grid_1D_delay
    from beccatest.worlds.grid_1D_ms import World as World_grid_1D_ms
    from beccatest.worlds.grid_1D_noise import World as World_grid_1D_noise
    from beccatest.worlds.grid_2D import World as World_grid_2D
    from beccatest.worlds.grid_2D_dc import World as World_grid_2D_dc
    from beccatest.worlds.image_1D import World as World_image_1D
    from beccatest.worlds.image_2D import World as World_image_2D
    from beccatest.worlds.fruit import World as World_fruit

    start_time = time.time()
    performance = []
    performance.append(test_world(World_grid_1D))
    performance.append(test_world(World_grid_1D_chase))
    performance.append(test_world(World_grid_1D_delay))
    performance.append(test_world(World_grid_1D_ms))
    performance.append(test_world(World_grid_1D_noise))
    performance.append(test_world(World_grid_2D))
    performance.append(test_world(World_grid_2D_dc))
    performance.append(test_world(World_image_1D))
    performance.append(test_world(World_image_2D))
    performance.append(test_world(World_fruit))
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
    print('Test suite completed in {0:.2} seconds'.format(
        finish_time - start_time))

    # Block the program, displaying all plots.
    # When the plot windows are closed, the program closes.
    plt.show()


def test_world(world_class, testing_period=TEST_LENGTH):
    """
    Test the brain's performance on a world.

    Parameters
    ----------
    world_class : World
        The class containing the BECCA-compatible world that the
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
    world = world_class(lifespan=testing_period)
    performance = run(world)
    print('Performance is: {0:.3}'.format(performance))
    return performance, world.name


def profile(World):
    """
    Profile the brain's performance on the selected world.
    """
    profiling_lifespan = 1e4
    print('Profiling BECCA\'s performance...')
    command = 'run(World(lifespan={0}), restore=True)'.format(
        profiling_lifespan)
    cProfile.run(command, 'becca_test.profile')
    profile_stats = pstats.Stats('becca_test.profile')
    profile_stats.strip_dirs().sort_stats('time', 'cumulative').print_stats(30)
    print('   View at the command line with')
    print(' > python -m pstats becca_test.profile')


def run(world, restore=False):
    """
    Run BECCA with a world.

    Connects the brain and the world together and runs them for as long
    as the world dictates.

    Parameters
    ----------
    restore : bool, optional
        If ``restore`` is True, try to restore the brain from a previously saved
        version, picking up where it left off.
        Otherwise it create a new one. The default is False.

    Returns
    -------
    performance : float
        The performance of the brain over its lifespan, measured by the
        average reward it gathered per time step.
    """
    start_time = time.time()
    brain_name = '{0}_brain'.format(world.name)
    brain = Brain(world.num_sensors, world.num_actions, brain_name=brain_name)
    if restore:
        brain = brain.restore()
    # Start at a resting state.
    actions = np.zeros(world.num_actions)
    sensors, reward = world.step(actions)
    # Repeat the loop through the duration of the existence of the world:
    # sense, act, repeat.
    while world.is_alive():
        actions = brain.sense_act_learn(sensors, reward)
        sensors, reward = world.step(actions)
        world.visualize(brain)
    performance = brain.report_performance()
    finish_time = time.time()
    print(world.name, 'ran in {0:.2} seconds'.format(finish_time - start_time))
    return performance


if __name__ == '__main__':
    # Build the command line parser.
    parser = argparse.ArgumentParser(
        description='Test BECCA on some toy worlds.')
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
    parser.add_argument('-p', '--profile', action='store_true')
    args = parser.parse_args()

    if args.world == 'grid1D' or args.world == '1': 
        from beccatest.worlds.grid_1D import World as World_grid_1D
        World = World_grid_1D
    elif args.world == 'grid1D_chase' or args.world == '2':
        from beccatest.worlds.grid_1D_chase import World as World_grid_1D_chase
        World = World_grid_1D_chase
    elif args.world == 'grid1D_delay' or args.world == '3':
        from beccatest.worlds.grid_1D_delay import World as World_grid_1D_delay
        World = World_grid_1D_delay
    elif args.world == 'grid1D_ms' or args.world == '4':
        from beccatest.worlds.grid_1D_ms import World as World_grid_1D_ms
        World = World_grid_1D_ms
    elif args.world == 'grid1D_noise' or args.world == '5':
        from beccatest.worlds.grid_1D_noise import World as World_grid_1D_noise
        World = World_grid_1D_noise
    elif args.world == 'grid2D' or args.world == '6':
        from beccatest.worlds.grid_2D import World as World_grid_2D
        World = World_grid_2D
    elif args.world == 'grid2D_dc' or args.world == '7':
        from beccatest.worlds.grid_2D_dc import World as World_grid_2D_dc
        World = World_grid_2D_dc
    elif args.world == 'image1D' or args.world == '8':
        from beccatest.worlds.image_1D import World as World_image_1D
        World = World_image_1D
    elif args.world == 'image2D' or args.world == '9':
        from beccatest.worlds.image_2D import World as World_image_2D
        World = World_image_2D
    elif args.world == 'fruit' or args.world == '10':
        from beccatest.worlds.fruit import World as World_fruit
        World = World_fruit
    else:
        args.world = 'all'

    if args.world == 'all':
        suite()
    elif args.profile:
        profile(World)
    else:
        performance = run(World(lifespan=1e6), restore=True)
        print('performance:', performance)
