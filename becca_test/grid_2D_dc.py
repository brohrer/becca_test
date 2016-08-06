"""
Decoupled two-dimensional grid task.

This is just like the regular 2D grid task, except that it the rows
and columns are sensed separately. This makes the task challenging.
Both the row number and column number need to be taken into
account in order to know what actions to take. This task requires building
basic sensory data into more complex features in order to do well.
"""
from __future__ import print_function
import numpy as np

import becca.connector
from becca_test.grid_2D import World as Grid_2D_World


class World(Grid_2D_World):
    """
    Decoupled two-dimensional grid world.

    It's just like the grid_2D world except that the sensors
    array represents a row and a column separately,
    rather than coupled together.

    Optimal performance is a reward of about 90 per time step.

    Attributes
    ----------
    See grid_2D.py for a full description of attributes.
    """
    def __init__(self, lifespan=None):
        """
        Set up the world based on the grid_2D world.

        Parameters
        ----------
        lifespan : int
            The number of time steps to continue the world.
        """
        Grid_2D_World.__init__(self, lifespan)
        self.name = 'grid_2D_dc'
        self.name_long = 'decoupled two dimensional grid world'
        print(", decoupled")
        self.num_sensors = self.world_size * 2


    def assign_sensors(self):
        """
        Create an appropriate sensor array

        Returns
        -------
        sensors : list of floats
            The current state of the world, reflected in the sensors.
        """
        sensors = np.zeros(self.num_sensors)
        # Sensors 0-4 represent each of the 5 rows.
        sensors[int(self.world_state[0])] = 1
        # Sensors 5-9 represent each of the 5 columns.
        sensors[int(self.world_state[1] + self.world_size)] = 1
        return sensors


if __name__ == "__main__":
    becca.connector.run(World())
