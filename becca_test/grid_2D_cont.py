"""
Two-dimensional grid task, with continuous sensor value.

This task is similar to grid_2D, but instead of providing
discrete sensor values, it provides two continuous ones.
In order to succesfully learn this task, becca must first
discretize the input.

To run this world from the command line

    python -m grid_2D_cont

"""
from __future__ import print_function

import becca.connector
from becca_test.grid_2D import World as Grid_2D_World


class World(Grid_2D_World):
    """
    Two-dimensional grid task with continous sensing.

    This task is identical to Grid_2D, with the exception
    that sensed position is returned as two floats,
    rather than one-hot discretized arrays.
    """
    def __init__(self, lifespan=None):
        Grid_2D_World.__init__(self, lifespan)
        self.name = 'grid_2D_continuous'
        self.name_long = 'two dimensional grid world, continuous sensors'
        print('  -- continuous sensors')
        self.num_sensors = 2
        self.visualize_interval = 1e3
        self.brain_visualize_interval = 1e4

    def sense(self):
        """
        Generate the appropriate sensor values for the current state.
        """
        return self.world_state


if __name__ == "__main__":
    becca.connector.run(World())
