"""
Runs the OpenAI cartpole environment using Becca.
"""

import becca.connector
from becca.base_world import World as BaseWorld

import gym
# import time
import numpy as np


class Cartpole(BaseWorld):
    """
    The cartpole environment from OpenAI.
    Info: https://gym.openai.com/envs/CartPole-v1/
    """


    def __init__(self, lifespan):

        BaseWorld.__init__(self, lifespan)
        self.name = 'cartpole_v1'
        self.name_long = 'cartpole_v1'
        print("Entering", self.name_long)

        # Number of elements from environment and actions available in
        # envrionment
        self.num_sensors = 4
        self.num_actions = 2

        # Make dat gym env
        self.env = gym.make('CartPole-v1')

        # Initialize sensors to initial value of observation from envrionment
        self.sensors = self.env.reset()

        self.action = np.zeros(self.num_actions)

        # Cartpole_counter is the number of steps within a full cartpole
        # simulation
        # num_iterations is the number of total cartpole iterations
        self.cartpole_counter = 0
        # self.num_iterations = 0

        # The world starts with no iterations
        self.timestep = 0

        # The number of full iterations of CartPole to run
        self.max_iterations = lifespan

        # Initialize world to be alive
        self.alive = True

        # Initialize world to be not done
        self.done = False

        # Set these to very low numbers for debugging
        self.world_visualize_period = 50
        self.brain_visualize_period = 50


    def step(self, actions):
        """
        Advance the world one time step.

        Parameters
        ----------
        actions : array of floats
            An array corresponding to the actions that can be chosen. The
            argmax of this array will be the chosen action.

        Returns
        -------
        reward : float
            The amount of reward or punishment given by the world.
        sensors : array of floats
            The values of each of the sensors.
        """

        # Choose an action; if both actions have the same value, randomly choose
        self.action = actions

        action = np.random.choice(np.flatnonzero(actions == self.action.max()))

        # Take a step in the environment, increment "timestep"
        self.sensors, self.reward, self.done, _ = self.env.step(action)
        self.cartpole_counter += 1

        # If we are "done"
        if self.done:
            # Increment the number of iterations
            # self.num_iterations += 1
            self.timestep += 1

            # Print how many steps this iteration lasted (max = 500)
            print("Iteration", self.timestep, "lasted",
                  self.cartpole_counter, "steps.")

            # Reset environment
            self.sensors = self.env.reset()
            self.cartpole_counter = 0
            self.done = False


        # If we have reached max_iterations, stop the environment
        if self.timestep == self.max_iterations:
            self.alive = False

        if self.cartpole_counter == 500:
            print("Won the game")
            self.alive = False

        return self.sensors, self.reward


    def is_alive(self):
        return self.alive


# def test():




if __name__ == '__main__':
    # start_time = time.time()
    becca.connector.run(Cartpole(lifespan=100))
    # finish_time = time.time()
    # delta_time = finish_time - start_time
    # print('Performance is: {0:.3}'.format(performance))
    # print(world.name, 'ran in {0:.2} seconds ({1:.2} minutes),'.format(
    #     delta_time, delta_time / 60.))
    # return performance, world.name
