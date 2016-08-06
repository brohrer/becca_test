"""
One-dimensional grid delay task.

This task tests an agent's ability to properly ascribe reward to the
correct cause. The reward is delayed by a variable amount, which
makes the task challenging.
"""
from __future__ import print_function
import numpy as np

import becca.connector
from becca_test.grid_1D import World as Grid_1D_World


class World(Grid_1D_World):
    """
    One-dimensional grid task with delayed reward

    This task is identical to the grid_1D task with the
    exception that reward is randomly delayed a few time steps.

    Most of this world's attributes are defined in base_world.py.
    The few that aren't are defined below.
    """
    def __init__(self, lifespan=None):
        """
        Initialize the world. Base it on the grid_1D world.

        Parameters
        ----------
        lifespan : int
            The number of time steps to continue the world.
        """
        Grid_1D_World.__init__(self, lifespan)
        self.name = 'grid_1D_delay'
        self.name_long = 'one dimensional grid world with delay'
        print('--delayed')

        # max_delay : int
        #     The maximum number of time steps that the reward may be delayed.
        self.max_delay = 1
        # future_reward : list of floats
        #     The reward that has been received, but will not be delivered to
        #     the agent yet. The index of the list indicates how many time
        #     steps will pass before delivery occurs.
        self.future_reward = [0.] * self.max_delay

        self.world_visualize_period = 1e6


    def assign_reward(self, sensors):
        """
        Calcuate the reward corresponding to the current state and assign
        it to a future time step.

        Parameters
        ----------
        sensors : array of floats
            The current sensor values.

        Returns
        -------
        reward : float
            The reward associated the set of input sensors.
        """
        new_reward = -sensors[8]
        new_reward += sensors[3]
        # Punish actions just a little
        new_reward -= self.energy  * self.energy_cost
        # Find the delay for the reward
        delay = np.random.randint(0, self.max_delay)
        self.future_reward[delay] += new_reward
        # Advance the reward future by one time step
        self.future_reward.append(0.)
        reward = self.future_reward.pop(0)
        return reward


    def visualize_world(self, brain):
        """
        Show what's going on in the world.
        """
        state_image = ['.'] * (self.num_sensors + self.num_actions + 2)
        state_image[self.simple_state] = 'O'
        state_image[self.num_sensors:self.num_sensors + 2] = '||'
        action_index = np.where(self.action > 0.1)[0]
        if action_index.size > 0:
            for i in range(action_index.size):
                state_image[self.num_sensors + 2 + action_index[i]] = 'x'
        print(''.join(state_image))


if __name__ == "__main__":
    becca.connector.run(World())
