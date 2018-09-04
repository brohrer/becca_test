"""
Vacuum cleaner world.

This task is inspired by Russell and Norvig's vacuum cleaner world.
http://web.ntnu.edu.tw/~tcchiang/ai/Vacuum%20Cleaner%20World.htm
It's purpose is to be a simple-as-possible world with an obvious
optimal policy for debugging.

Usage
To run this world standalone from the command line

    python -m vacuum
"""
from __future__ import print_function
import numpy as np

import becca.connector
from becca.base_world import World as BaseWorld


class World(BaseWorld):
    """
    In this task, a two-room house needs to get clean.
    Room A (0) is on the left and Room B (1) is on the right.
    The vacuum has two actions it can choose: move Left (0) or Right (1).
    When in Room A, moving Right gets the robot into Room B.
    When in Room B, moving Left gets the robot into Room A.
    Moving into a room gains the robot a reward of 1
    (This is based on the empirical observation that as soon as one
    leaves a clean room, it becomes instantly dirty again.)
    Running into a wall hurts and gets the robot a reward of -1
    (a stiff punishment).

    To get the most reward possible, the robot should alternate
    Right and Left actinos.

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
        self.name = 'vacuum'
        self.name_long = 'vacuum cleaner world'
        print("Entering", self.name_long)

        self.num_sensors = 2
        self.n_positions = self.num_sensors
        self.num_actions = 2
        # Left: 0
        # Right: 1
        self.action = np.zeros(self.num_actions)
        # The room the robot is in.
        # Room A: 0
        # Room B: 1
        self.state = 0

        self.visualize_interval = 1e3
        self.brain_visualize_interval = 1e3

    def step(self, action):
        """
        Advance the world one time step.

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

        reward = 0
        old_state = self.state
        if self.action[0]:
            self.state -= 1
        if self.action[1]:
            self.state += 1
        # Check for collisions.
        if self.state == -1:
            reward = -1
            self.state = 0
        if self.state == 2:
            reward = -1
            self.state = 1

        # Check for a room change.
        if np.abs(self.state - old_state) == 1:
            reward = 1

        sensors = np.zeros(self.num_sensors)
        sensors[int(self.state)] = 1

        return sensors, reward

    def visualize(self):
        """
        Show what's going on in the world.
        """
        state_image = ['.'] * (self.num_sensors + self.num_actions + 2)
        state_image[int(self.state)] = 'O'
        state_image[self.num_sensors:self.num_sensors + 2] = '||'
        action_index = np.where(self.action > 0.1)[0]
        if action_index.size > 0:
            for i in range(action_index.size):
                state_image[self.num_sensors + 2 + action_index[i]] = 'x'
        print(''.join(state_image))


if __name__ == "__main__":
    becca.connector.run(World())
