"""
One-dimensional grid task in which the target moves.

This task tests an brain's ability to choose an appropriate action.
It is straightforward. Reward and punishment is clear and immediate.
There is only one reward state and it can be reached in a single
step. The target keeps moving, so it does require the ability to respond
to sensory information.
"""
from __future__ import print_function
import numpy as np

from beccatest.base_world import World as BaseWorld


class World(BaseWorld):
    """
    One-dimensional grid world with a moving target.

    In this task, the brain steps forward and backward along
    a five-position line. The one position is rewarded. Each time
    BECCA reaches the target, it jumps to a new position.
    There is also a slight
    punishment for effort expended in taking actions.
    This is intended to be a simple
    task for troubleshooting BECCA on the full ``becca_world_chase`` task.

    Attributes
    ----------
    action : array of float
        The most recent set of action commands received.
    brain_visualize_period : int
        The number of time steps between creating a full visualization of
        the ``brain``.
    energy_cost : float
        The punishment per position step taken.
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
    position : int
        The position of the agent in the world.
    reward_magnitude : float
        The magnitude of the reward and punishment given at
        rewarded or punished positions.
    target_position : int
        The position of the target in the world.
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
        self.energy_cost = self.reward_magnitude / 1e2
        self.name = 'grid_1D_chase'
        self.name_long = 'one dimensional chase grid world'
        print("Entering", self.name_long)
        self.size = 7
        self.num_sensors = self.size + 2 * (self.size - 1)
        self.num_actions = 2 * (self.size - 1)
        self.reward = 0.
        self.energy = 0.
        self.action = np.zeros(self.num_actions)
        self.sensors = np.zeros(self.num_sensors)
        self.position = 2
        self.target_position = 1
        self.world_visualize_period = 1e3
        self.brain_visualize_period = 1e3


    def step(self, action):
        """
        Advance the world one time step.

        Parameters
        ----------
        action : array of floats
            The set of action commands to execute.

        Returns
        -------
        self.reward : float
            The amount of reward or punishment given by the world.
        self.sensors : array of floats
            The values of each of the sensors.
        """
        self.action = action
        self.action = np.round(self.action)
        self.timestep += 1

        # Find the step size as combinations of the action commands.
        # For world of size 5:
        #     action[i]     result
        #            0      1 step to the right
        #            1      2 steps to the right
        #            2      3 steps to the right
        #            3      4 steps to the right
        #            4      1 step to the left
        #            5      2 steps to the left
        #            6      3 steps to the left
        #            7      4 steps to the left
        scale = np.arange(self.size - 1) + 1.
        step_size = (np.sum(self.action[:self.size - 1] * scale) -
                     np.sum(self.action[self.size - 1:
                                        2 * (self.size - 1)] * scale))
        # Action cost is an approximation of metabolic energy.
        # Action cost is proportional to the number of steps taken.
        self.energy = (np.sum(self.action[:self.size - 1] * scale) +
                       np.sum(self.action[self.size - 1:
                                          2 * (self.size - 1)] * scale))

        self.position += step_size
        self.position = np.minimum(self.position, self.size - 1)
        self.position = int(np.maximum(self.position, 0))
        self.assign_reward()
        self.move_target()

        self.sensors = np.zeros(self.num_sensors)
        # Sense the agent's presence in each bin.
        self.sensors[self.position] = 1
        # Sense the relative distance to the target.
        distance = self.position - self.target_position
        if distance < 0:
            self.sensors[self.size - 1 + np.abs(distance)] = 1
        else:
            self.sensors[2 * (self.size - 1) + np.abs(distance)] = 1

        return self.sensors, self.reward


    def move_target(self):
        """
        Move the target to a new position that is not already occupied.
        """
        while self.target_position == self.position:
            self.target_position = int(np.random.randint(self.size))


    def assign_reward(self):
        """
        Calculate the total reward corresponding to the current state
        """
        self.reward = 0.
        if self.position == self.target_position:
            self.reward += self.reward_magnitude
        # Punish actions just a little
        self.reward -= self.energy  * self.energy_cost


    def visualize_world(self, brain):
        """
        Show what's going on in the world.
        """
        state_image = ['.'] * (self.size + self.num_actions + 2)
        state_image[self.position] = 'O'
        state_image[self.target_position] = '+'
        state_image[self.size:self.size + 2] = '||'
        action_index = np.where(self.action > 0.1)[0]
        if action_index.size > 0:
            for i in range(action_index.size):
                state_image[self.size + 2 + action_index[i]] = 'x'
        state_string = ''.join(state_image)
        print(state_string, '  ', self.timestep, 'time steps')
