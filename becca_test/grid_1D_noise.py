"""
One-dimensional grid task with noise

In this task, the agent has the challenge of discriminating between
actual informative state sensors, and a comparatively large number
of sensors that are pure noise distractors. Many learning methods
make the implicit assumption that all sensors are informative.
This task is intended to break them.
"""
from __future__ import print_function
import numpy as np

import becca.connector
from becca_test.base_world import World as BaseWorld


class World(BaseWorld):
    """
    One-dimensional grid world with noise.

    In this world, the agent steps forward and backward
    along three positions on a line. The second position is rewarded
    and the first and third positions are punished. Also, any actions
    are penalized somewhat. It also includes some inputs that are pure noise.
    Optimal performance is a reward of about .70 per time step.

    Most of this world's attributes are defined in base_world.py.
    The few that aren't are defined below.
    """
    def __init__(self, lifespan=None):
        """
        Set up the world.
        """
        BaseWorld.__init__(self, lifespan)
        self.name = 'grid_1D_noise'
        self.name_long = 'noisy one dimensional grid world'
        print("Entering", self.name_long)
        self.num_real_sensors = 3
        # num_noise_sensors : int
        #     Of the sensors, these are purely noise.
        #     These have no basis in the world and are only meant to distract.
        # num_real_sensors : int
        #     Of the sensors, these are the ones that represent position.
        self.num_noise_sensors = 0
        self.num_sensors = self.num_noise_sensors + self.num_real_sensors
        self.num_actions = 2
        self.action = np.zeros(self.num_actions)
        # energy_cost : float
        #     The punishment per position step taken.
        self.energy_cost = 0.01
        # jump_fraction : float
        #     The fraction of time steps on which the agent jumps to
        #     a random position.
        self.jump_fraction = 0.1

        # world_state : float
        #     The actual position of the agent in the world.
        #     This can be fractional.
        self.world_state = 0
        # simple_state : int
        #     The nearest integer position of the agent in the world.
        self.simple_state = 0

        self.world_visualize_period = 1e6
        self.brain_visualize_period = 1e3


    def step(self, action):
        """
        Take one time step through the world

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
        self.action = action.copy().ravel()
        self.action[np.nonzero(self.action)] = 1.
        self.timestep += 1
        step_size = self.action[0] - self.action[1]

        # An approximation of metabolic energy.
        energy = self.action[0] + self.action[1]
        self.world_state = self.world_state + step_size

        # At random intervals, jump to a random position in the world.
        if np.random.random_sample() < self.jump_fraction:
            self.world_state = (self.num_real_sensors *
                                np.random.random_sample())

        # Ensure that the world state falls between 0 and num_real_sensors.
        self.world_state -= (self.num_real_sensors *
                             np.floor_divide(self.world_state,
                                             self.num_real_sensors))
        self.simple_state = int(np.floor(self.world_state))

        # Assign sensors as zeros or ones.
        # Represent the presence or absence of the current position in the bin.
        real_sensors = np.zeros(self.num_real_sensors)
        real_sensors[self.simple_state] = 1

        # Generate a set of noise sensors
        noise_sensors = np.round(np.random.random_sample(
            self.num_noise_sensors))
        sensors = np.hstack((real_sensors, noise_sensors))
        reward = -1.
        if self.simple_state == 1:
            reward = 1.
        reward -= energy * self.energy_cost

        return sensors, reward


    def visualize_world(self, brain):
        """
        Show what's going on in the world.
        """
        state_image = ['.'] * (self.num_real_sensors +
                               self.num_actions + 2)
        state_image[self.simple_state] = 'O'
        state_image[self.num_real_sensors:self.num_real_sensors + 2] = '||'
        action_index = np.where(self.action > 0.1)[0]
        if action_index.size > 0:
            state_image[self.num_real_sensors + 2 + action_index[0]] = 'x'
        print(''.join(state_image))


if __name__ == "__main__":
    becca.connector.run(World())
