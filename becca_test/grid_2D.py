"""
Two-dimensional grid task.

This task is a 2D extension of the 1D grid world and
is similar to it in many ways. It is a little more
challenging, because can take two actions to reach a reward state.
"""
from __future__ import print_function
import numpy as np

import becca.connector
from becca_test.base_world import World as BaseWorld


class World(BaseWorld):
    """
    Two-dimensional grid world.

    In this world, the agent steps North, South, East, or West in
    a 5 x 5 grid-world. Position (4,4) is rewarded and (2,2)
    is punished. There is also a lesser penalty for each
    horizontal or vertical step taken.
    Optimal performance is a reward of about .9 per time step.

    Some of this world's attributes are defined in base_world.py.
    The others are defined below.
    """
    def __init__(self, lifespan=None):
        """
        Initialize the world.

        Parameters
        ----------
        lifespan : int
            The number of time steps to continue the world.
        """
        BaseWorld.__init__(self, lifespan)
        self.name = 'grid_2D'
        self.name_long = 'two dimensional grid world'
        print("Entering", self.name_long)

        self.num_actions = 8
        # world_size : int
        #     The world consists of a 2D grid of size
        #     world_size by world_size.
        self.world_size = 5
        self.num_sensors = self.world_size ** 2
        # world_state : float
        #     The actual position of the agent in the world.
        #     This can be fractional.
        self.world_state = np.array([1., 1.])
        # targets : list of tuples of ints
        #     Each tuple is a (row, column) pair indicating a location
        #     that is rewarded.
        #     Reward positions (2,2) and (4,4)
        self.targets = [(1, 1), (3, 3)]
        self.action = np.zeros(self.num_actions)
        # energy_cost : float
        #     The punishment per position step taken.
        self.energy_cost = 0.05
        # jump_fraction : float
        #     The fraction of time steps on which the agent jumps to
        #     a random position.
        self.jump_fraction = 0.1
        # obstacles : list of tuples of ints
        #     Each tuple is a (row, column) pair indicating a location
        #     that are punished.
        #     Punish positions (2,4) and (4,2)
        self.obstacles = [(1, 3), (3, 1)]

        self.world_visualize_period = 1e6
        self.brain_visualize_period = 1e3


    def step(self, action):
        """
        Advance the world by one time step.

        Parameters
        ----------
        action : array of floats
            The set of action commands to execute.

        Returns
        -------
        reward : float
            The amount of reward or punishment given by the world.
        sensors : array of floats
            The values of each of the sensors.
        """
        # Turn the action command into a change in the world.
        self.action = action.ravel()
        self.action[np.nonzero(self.action)] = 1.
        self.timestep += 1
        self.world_state += (self.action[0:2] -
                             self.action[4:6] +
                             2 * self.action[2:4] -
                             2 * self.action[6:8]).T
        energy = (np.sum(self.action[0:2]) +
                  np.sum(self.action[4:6]) +
                  np.sum(2 * self.action[2:4]) +
                  np.sum(2 * self.action[6:8]))

        # At random intervals, jump to a random position in the world.
        if np.random.random_sample() < self.jump_fraction:
            self.world_state = (
                np.random.randint(0, self.world_size,
                                  size=len(self.world_state)).astype(float))

        # Enforce lower and upper limits on the grid world
        # by looping them around.
        self.world_state = np.remainder(self.world_state, self.world_size)
        sensors = self.assign_sensors()

        # Assign the reward appropriate to the current state.
        reward = 0.
        for obstacle in self.obstacles:
            if tuple(self.world_state) == obstacle:
                reward = -1.
        for target in self.targets:
            if tuple(self.world_state) == target:
                reward = 1.
        reward -= self.energy_cost * energy

        return sensors, reward


    def assign_sensors(self):
        """
        Construct the sensor array from the state information.

        Returns
        -------
        sensors : list of floats
            The current state of the world, reflected in the sensors.
        """
        sensors = np.zeros(self.num_sensors)
        sensors[int(self.world_state[0] +
                    self.world_state[1] * self.world_size)] = 1
        return sensors


    def visualize_world(self, brain):
        """
        Show the state of the world and the brain.
        """
        print(''.join(['state', str(self.world_state), '  action',
                       str((self.action[0:2] + 2 * self.action[2:4] -
                            self.action[4:6] - 2 * self.action[6:8]).T)]))


if __name__ == "__main__":
    becca.connector.run(World())
