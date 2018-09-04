"""
A multi-step variation on the one-dimensional grid task.

This is intended to be as similar as possible to the
one-dimensional grid task, but requires multi-step planning
or time-delayed reward assignment for optimal behavior.
"""
from __future__ import print_function
import numpy as np

import becca.connector
from becca.base_world import World as BaseWorld


class World(BaseWorld):
    """
    One-dimensional grid world, multi-step variation

    In this world, the agent steps forward and backward along a line.
    The fourth position is rewarded and the ninth position is punished.
    Optimal performance is a reward of about 85 per time step.

    Most of this world's attributes are defined in base_world.py.
    The few that aren't are defined below.
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
        self.name = 'grid_1D_ms'
        self.name_long = 'multi-step one dimensional grid world'
        print("Entering", self.name_long)

        self.num_sensors = 9
        self.num_positions = self.num_sensors
        self.num_actions = 2
        self.action = np.zeros(self.num_actions)
        self.energy = 0.
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
        self.simple_state = 1

        self.visualize_interval = 1e6

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
        self.action = action
        self.action = np.round(self.action)
        self.timestep += 1
        self.energy = self.action[0] + self.action[1]
        self.world_state += self.action[0] - self.action[1]
        # Occasionally add a perturbation to the action to knock it
        # into a different state.
        if np.random.random_sample() < self.jump_fraction:
            self.world_state = self.num_positions * np.random.random_sample()
        # Ensure that the world state falls between 0 and 9
        self.world_state -= self.num_positions * np.floor_divide(
            self.world_state, self.num_positions)
        self.simple_state = int(np.floor(self.world_state))
        if self.simple_state == 9:
            self.simple_state = 0
        sensors = self.sense()
        reward = self.assign_reward()
        return sensors, reward

    def sense(self):
        """
        Generate the appropriate sensor values for the current state.
        """
        # Assign sensors as zeros or ones.
        # Represent the presence or absence of the current position in the bin.
        sensors = np.zeros(self.num_sensors)
        sensors[self.simple_state] = 1
        return sensors

    def assign_reward(self):
        """
        Calculate the total reward corresponding to the current state

        Returns
        -------
        reward : float
            The reward associated the set of input sensors.
        """
        reward = 0.
        if int(self.world_state) == 3:
            reward += 1.
        if int(self.world_state) == 8:
            reward -= 1.
        # Punish actions just a little
        reward -= self.energy * self.energy_cost
        reward = np.maximum(reward, -1.)
        return reward

    def visualize(self, brain):
        """
        Show what's going on in the world.
        """
        state_image = ['.'] * (self.num_sensors + self.num_actions + 2)
        state_image[int(self.world_state)] = 'O'
        state_image[self.num_positions:self.num_positions + 2] = '||'
        action_index = np.where(self.action > 0.1)[0]
        if action_index.size > 0:
            for i in range(action_index.size):
                state_image[self.num_positions + 2 + action_index[i]] = 'x'
        print(''.join(state_image))


if __name__ == "__main__":
    becca.connector.run(World())
