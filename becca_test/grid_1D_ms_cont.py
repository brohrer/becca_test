"""
One-dimensional multi-step grid task, with continuous sensor value.

This task is similar to grid_1D_ms, but instead of providing
discrete sensor values, it provides a single continuous one.
In order to succesfully learn this task, becca must first
discretize the input.

To run this world from the command line

    python -m grid_1D_ms_cont

"""
from __future__ import print_function

import becca.connector
from becca_test.grid_1D_ms import World as Grid_1D_MS_World


class World(Grid_1D_MS_World):
    """
    One-dimentional multi-step grid task with continous sensing.

    This task is identical to grid_1D_ms, with the exception
    that sensed position is returned as a float,
    rather than a one-hot discretized array.
    """
    def __init__(self, lifespan=None):
        Grid_1D_MS_World.__init__(self, lifespan)
        self.name = 'grid_1D_ms_continuous'
        self.name_long = ('one dimensional multi-step grid world, ' +
                          'continuous sensor')
        print('  -- continuous sensor')
        self.num_sensors = 1
        self.visualize_interval = 1e6

    def sense(self):
        """
        Generate the appropriate sensor values for the current state.
        """
        return [self.world_state]


if __name__ == "__main__":
    becca.connector.run(World())
