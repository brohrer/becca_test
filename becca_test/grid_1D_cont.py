"""
One-dimensional grid task, with continuous sensor value.

This task is similar to grid_1D, but instead of providing
discrete sensor values, it provides a single continuous one.
In order to succesfully learn this task, becca must first
discretize the input.

To run this world from the command line

    python -m grid_1D_cont

"""
from __future__ import print_function

import becca.connector
from becca_test.grid_1D import World as Grid_1D_World


class World(Grid_1D_World):
    """
    One-dimentional grid task with continous sensing.

    This task is identical to Grid_1D, with the exception
    that sensed position is returned as a float,
    rather than a one-hot discretized array.
    """
    def __init__(self, lifespan=None):
        Grid_1D_World.__init__(self, lifespan)
        self.name = 'grid_1D_continuous'
        self.name_long = 'one dimensional grid world, continuous sensor'
        print('  -- continuous sensor')
        self.num_sensors = 1
        self.visualize_interval = 1e3
        self.brain_visualize_interval = 1e4

    def sense(self):
        """
        Generate the appropriate sensor values for the current state.
        """
        return [self.world_state]


if __name__ == "__main__":
    becca.connector.run(World())
