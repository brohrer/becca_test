"""
Two-dimensional grid task.

This task is a 2D extension of the 1D grid world and
is similar to it in many ways. It is a little more
challenging, because can take two actions to reach a reward state.
"""
from __future__ import print_function
import numpy as np
from beccatest.base_world import World as BaseWorld


class World(BaseWorld):
    """
    Two-dimensional grid world.

    In this world, the agent steps North, South, East, or West in
    a 5 x 5 grid-world. Position (4,4) is rewarded and (2,2)
    is punished. There is also a lesser penalty for each
    horizontal or vertical step taken.
    Optimal performance is a reward of about .9 per time step.

    Attributes
    ----------
    action : array of floats
        The most recent set of action commands received.
    brain_visualize_period : int
        The number of time steps between creating a full visualization of
        the ``brain``.
    energy_cost : float
        The punishment per position step taken.
    jump_fraction : float
        The fraction of time steps on which the agent jumps to
        a random position.
    name : str
        A name associated with this world.
    name_long : str
        A longer name associated with this world.
    num_actions : int
        The number of action commands this world expects. This should be
        the length of the action array received at each time step.
    num_sensors : int
        The number of sensor values the world returns to the brain
        at each time step.
    obstacles : list of tuples of ints
        Each tuple is a (row, column) pair indicating a location
        that are punished.
    reward_magnitude : float
        The magnitude of the reward and punishment given at
        rewarded or punished positions.
    targets : list of tuples of ints
        Each tuple is a (row, column) pair indicating a location
        that is rewarded.
    world_size : int
        The world consists of a 2D grid of size
        ``world_size`` by ``world_size``.
    world_state : float
        The actual position of the agent in the world. This can be fractional.
    world_visualize_period : int
        The number of time steps between creating visualizations of
        the world.
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
        self.reward_magnitude = 1.
        self.energy_cost = 0.05 * self.reward_magnitude
        self.jump_fraction = 0.1
        self.name = 'grid_2D'
        self.name_long = 'two dimensional grid world'
        print("Entering", self.name_long)
        self.num_actions = 8
        self.world_size = 5
        self.num_sensors = self.world_size ** 2
        self.world_state = np.array([1., 1.])
        # Reward positions (2,2) and (4,4)
        self.targets = [(1, 1), (3, 3)]
        # Punish positions (2,4) and (4,2)
        self.action = np.zeros(self.num_actions)
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
                                  size=self.world_state.size).astype(float))

        # Enforce lower and upper limits on the grid world
        # by looping them around.
        self.world_state = np.remainder(self.world_state, self.world_size)
        sensors = self.assign_sensors()

        # Assign the reward appropriate to the current state.
        reward = 0.
        for obstacle in self.obstacles:
            if tuple(self.world_state) == obstacle:
                reward = - self.reward_magnitude
        for target in self.targets:
            if tuple(self.world_state) == target:
                reward = self.reward_magnitude
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
        Show the state of the world and the ``brain``.
        """
        print(''.join(['state', str(self.world_state), '  action',
                       str((self.action[0:2] + 2 * self.action[2:4] -
                            self.action[4:6] - 2 * self.action[6:8]).T)]))
