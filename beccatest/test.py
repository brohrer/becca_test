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

from becca.brain import Brain


def suite(lifespan=1e4):
    """
    Run all the worlds in the benchmark and tabulate their performance.
    """
    # Import the suite of test worlds
    from beccatest.grid_1D import World as World_grid_1D
    from beccatest.grid_1D_chase import World as World_grid_1D_chase
    from beccatest.grid_1D_delay import World as World_grid_1D_delay
    from beccatest.grid_1D_ms import World as World_grid_1D_ms
    from beccatest.grid_1D_noise import World as World_grid_1D_noise
    from beccatest.grid_2D import World as World_grid_2D
    from beccatest.grid_2D_dc import World as World_grid_2D_dc
    from beccatest.image_1D import World as World_image_1D
    from beccatest.image_2D import World as World_image_2D
    from beccatest.fruit import World as World_fruit

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
    start_time = time.time()
    world = world_class(lifespan=lifespan)
    performance = run(world)
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
    print('Profiling BECCA\'s performance...')
    command = 'run(World(lifespan={0}), restore=True)'.format(lifespan)
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
    parser.add_argument(
        '-p', '--profile', action='store_true',
        help="Profile BECCA's performance.")
    parser.add_argument(
        '-t', '--lifespan', type=int, 
        help='The number of time steps (in thousands) to run the world.')
    args = parser.parse_args()

    if args.world == 'grid1D' or args.world == '1': 
        from beccatest.grid_1D import World as World_grid_1D
        World = World_grid_1D
    elif args.world == 'grid1D_chase' or args.world == '2':
        from beccatest.grid_1D_chase import World as World_grid_1D_chase
        World = World_grid_1D_chase
    elif args.world == 'grid1D_delay' or args.world == '3':
        from beccatest.grid_1D_delay import World as World_grid_1D_delay
        World = World_grid_1D_delay
    elif args.world == 'grid1D_ms' or args.world == '4':
        from beccatest.grid_1D_ms import World as World_grid_1D_ms
        World = World_grid_1D_ms
    elif args.world == 'grid1D_noise' or args.world == '5':
        from beccatest.grid_1D_noise import World as World_grid_1D_noise
        World = World_grid_1D_noise
    elif args.world == 'grid2D' or args.world == '6':
        from beccatest.grid_2D import World as World_grid_2D
        World = World_grid_2D
    elif args.world == 'grid2D_dc' or args.world == '7':
        from beccatest.grid_2D_dc import World as World_grid_2D_dc
        World = World_grid_2D_dc
    elif args.world == 'image1D' or args.world == '8':
        from beccatest.image_1D import World as World_image_1D
        World = World_image_1D
    elif args.world == 'image2D' or args.world == '9':
        from beccatest.image_2D import World as World_image_2D
        World = World_image_2D
    elif args.world == 'fruit' or args.world == '10':
        from beccatest.fruit import World as World_fruit
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
        performance = run(World(lifespan=lifespan), restore=True)
        print('performance:', performance)
