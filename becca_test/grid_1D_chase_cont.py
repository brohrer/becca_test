"""
One-dimensional grid task with moving target, with continuous sensor value.

This task is similar to grid_1D_chase, but instead of providing
discrete sensor values, it provides a single continuous one.
In order to succesfully learn this task, becca must first
discretize the input.

To run this world from the command line

    python -m grid_1D_chase_cont

"""
from __future__ import print_function

import becca.connector
from becca_test.grid_1D_chase import World as Grid_1D_Chase_World


class World(Grid_1D_Chase_World):
    """
    One-dimentional moving target grid task with continous sensing.

    This task is identical to Grid_1D_Chase, with the exception
    that sensed position and distance are returned as floats.
    """
    def __init__(self, lifespan=None):
        Grid_1D_Chase_World.__init__(self, lifespan)
        self.name = 'grid_1D_chase_continuous'
        self.name_long = 'one dimensional chase grid world, continuous sensor'
        print('  -- continuous sensor')
        self.num_sensors = 2
        self.visualize_interval = 1e6

    def sense(self):
        """
        Represent the world's internal state as an array of sensors.

        Returns
        -------
        array of floats
            The set of sensor values.
        """
        distance = self.position - self.target_position
        return (self.position, distance)


if __name__ == "__main__":
    becca.connector.run(World())
