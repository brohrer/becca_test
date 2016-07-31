"""
The base world on which all the other worlds are based.
"""
from __future__ import print_function
import numpy as np


class World(object):
    """
    The base class for creating a new world.

    Attributes
    ----------
    brain_visualization_period : int
        The interval between rendering the state of the brain for the user..
    lifespan : float
        The number of time steps for which the world should remain active.
    name : String
        The name of the world.
    num_actions, num_sensors : int
        The number of actions and sensors, respectively.
    timestep : int
        The number of time steps that the world has already been through.
    world_visualization_period : int
        The interval between rendering the state of the world for the user.
    """
    def __init__(self, lifespan=None):
        """
        Initialize a new world with some benign default values.

        Parameters
        ----------
        lifespan : int or None
            The number of time steps that the world will be
            allowed to continue.
        """
        if lifespan is None:
            self.lifespan = 10 ** 4
        else:
            self.lifespan = lifespan
        # Starting at -1 allows for an intialization pass.
        self.timestep = -1
        self.world_visualize_period = 1e4
        self.brain_visualize_period = 1e4
        self.name = 'abstract base world'
        # These will likely be overridden in any subclass
        self.num_sensors = 0
        self.num_actions = 0


    def step(self, actions):
        """
        Take a timestep through an empty world that does nothing.

        Parameters
        ----------
        actions : array of floats
            The set of actions that the world can be expected to execute.

        Returns
        -------
        sensors : array of floats
            The current values of each of those sensors in the world.
        reward : float
            The current reward provided by the world.
        """
        self.timestep += 1
        sensors = np.zeros(self.num_sensors)
        actions = np.zeros(actions.shape)
        reward = 0
        return sensors, reward


    def is_alive(self):
        """
        Check whether the world is alive.

        Returns
        -------
        If False, the world has come to an end.
        """
        return self.timestep < self.lifespan


    def visualize(self, brain):
        """
        Let the world show BECCA's internal state as well as its own.

        Parameters
        ----------
        brain : Brain
            A copy of the ``Brain``, provided to the world so that the
            world can interpret and visualize it in the context of the
            world.
        """
        if (self.timestep % self.world_visualize_period) == 0:
            self.visualize_world(brain)
        if (self.timestep % self.brain_visualize_period) == 0:
            brain.visualize()


    def visualize_world(self, brain):
        """
        Show the user the state of the world.
        """
        print('{0} is {1} time steps old.'.format(self.name, self.timestep))
        print('The brain is {0} time steps old.'.format(brain.timestep))

