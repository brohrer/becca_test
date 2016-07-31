"""
Two-dimensional visual servo task

Like the 1D visual servo task, this task gives BECCA a chance
to build a comparatively large number of sensors into
a few informative features.
"""
from __future__ import print_function
import os

import matplotlib.pyplot as plt
import numpy as np

from beccatest.base_world import World as BaseWorld
import beccatest.world_tools as wtools
MODPATH = os.path.dirname(os.path.abspath(__file__))


class World(BaseWorld):
    """
    Two-dimensional visual servo world

    In this world, BECCA can direct its gaze up, down, left, and
    right, saccading about an image_data of a black square on a white
    background. It is rewarded for directing it near the center.
    Optimal performance is a reward of around .8 reward per time step.

    Attributes
    ----------
    action : array of floats
        The most recent set of action commands received.
    block_width : int
        The width of each superpixel, in number of columns.
    brain_visualize_period : int
        The number of time steps between creating a full visualization of
        the ``brain``.
    column_history : list if ints
        A time series of the location (measured in column pixels) of the
        center of the brain's field of view.
    column_min, column_max : int
        The low and high bounds on where the field of view can be centered.
    column_position : int
        The current location of the center of the field of view.
    fov_fraction : float
        The approximate fraction of the height and width of the image
        that the field of view occupies.
    fov_height, fov_width : float
        The height and width (in number of pixel rows and columns)
        of the field of view.
    fov_span : int
        The world pixelizes its field of view into a superpixel array
        that is ``fov_span`` X ``fov_span``.
    image_data : array of floats
        The image, read in and stored as a 2D numpy array.
    image_filename : str
        The file name of the image including the relative path.
    jump_fraction : float
        The fraction of time steps on which the agent jumps to
        a random position.
    max_step_size : int
        The largest step size allowed, in pixels in the original image.
    name : str
        A name associated with this world.
    name_long : str
        A longer name associated with this world.
    noise_magnitude : float
        A scaling factor that drives how much inaccurate each movement
        will be.
    num_actions : int
        The number of action commands this world expects. This should be
        the length of the action array received at each time step.
    num_sensors : int
        The number of sensor values the world returns to the brain
        at each time step.
    reward_magnitude : float
        The magnitude of the reward and punishment given at
        rewarded or punished positions.
    reward_region_width : int
        The width of the region, in number of columns, within which
        the center of the field of view gets rewarded.
    row_history : list if ints
        A time series of the location (measured in row pixels) of the
        center of the brain's field of view.
    row_min, row_max : int
        The low and high bounds on where the field of view can be centered.
    row_position : int
        The current location of the center of the field of view.
    step_cost : float
        The punishment per position step taken.
    target_column, target_row : int
        The row and column index that marks the center of the rewarded region.
    world_visualize_period : int
        The number of time steps between creating visualizations of
        the world.

    """
    def __init__(self, lifespan=None):
        """
        Set up the world.

        Parameters
        ----------
        lifespan : int
            The number of time steps to continue the world.
        """
        BaseWorld.__init__(self, lifespan)
        self.reward_magnitude = 1.
        self.jump_fraction = .05
        self.name = 'image_2D'
        self.name_long = 'two dimensional visual world'
        print("Entering", self.name_long)
        self.fov_span = 5
        # Initialize the image_data to be used as the environment
        self.image_filename = os.path.join(MODPATH, 'images', 'block_test.png')
        self.image_data = plt.imread(self.image_filename)
        # Convert it to grayscale if it's in color
        if self.image_data.shape[2] == 3:
            # Collapse the three RGB matrices into one b/w value matrix
            self.image_data = np.sum(self.image_data, axis=2) / 3.0
        # Define the size of the field of view, its range of
        # allowable positions, and its initial position.
        (im_height, im_width) = self.image_data.shape
        im_size = np.minimum(im_height, im_width)
        self.max_step_size = im_size / 2
        self.target_column = im_width / 2
        self.target_row = im_height / 2
        self.reward_region_width = im_size / 8
        self.noise_magnitude = 0.1
        self.fov_fraction = 0.5
        self.fov_height = im_size * self.fov_fraction
        self.fov_width = self.fov_height
        self.column_min = int(np.ceil(self.fov_width / 2))
        self.column_max = int(np.floor(im_width - self.column_min))
        self.row_min = int(np.ceil(self.fov_height / 2))
        self.row_max = int(np.floor(im_height - self.row_min))
        self.column_position = np.random.random_integers(self.column_min,
                                                         self.column_max)
        self.row_position = np.random.random_integers(self.row_min,
                                                      self.row_max)
        self.num_sensors = 2 * self.fov_span ** 2
        self.num_actions = 16
        self.sensors = np.zeros(self.num_sensors)
        self.action = np.zeros(self.num_actions)
        self.reward = 0.
        self.column_history = []
        self.row_history = []
        self.world_visualize_period = 1e3
        self.brain_visualize_period = 1e3
        self.print_features = False
        # TODO: re-implment print features


    def step(self, action):
        """
        Advance the world by one time step.

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
        self.timestep += 1
        self.action = action.ravel()
        self.action[np.nonzero(self.action)] = 1.

        # Actions 0-3 move the field of view to a higher-numbered
        # row (downward in the image_data) with varying magnitudes,
        # and actions 4-7 do the opposite.
        # Actions 8-11 move the field of view to a higher-numbered
        # column (rightward in the image_data) with varying magnitudes,
        # and actions 12-15 do the opposite.
        row_step = np.round(action[0] * self.max_step_size / 2 +
                            action[1] * self.max_step_size / 4 +
                            action[2] * self.max_step_size / 8 +
                            action[3] * self.max_step_size / 16 -
                            action[4] * self.max_step_size / 2 -
                            action[5] * self.max_step_size / 4 -
                            action[6] * self.max_step_size / 8 -
                            action[7] * self.max_step_size / 16)
        column_step = np.round(action[8] * self.max_step_size / 2 +
                               action[9] * self.max_step_size / 4 +
                               action[10] * self.max_step_size / 8 +
                               action[11] * self.max_step_size / 16 -
                               action[12] * self.max_step_size / 2 -
                               action[13] * self.max_step_size / 4 -
                               action[14] * self.max_step_size / 8 -
                               action[15] * self.max_step_size / 16)

        row_step = np.round(row_step * (
            1. + np.random.normal(scale=self.noise_magnitude)))
        column_step = np.round(column_step * (
            1. + np.random.normal(scale=self.noise_magnitude)))
        self.row_position = self.row_position + int(row_step)
        self.column_position = self.column_position + int(column_step)

        # Respect the boundaries of the image_data
        self.row_position = max(self.row_position, self.row_min)
        self.row_position = min(self.row_position, self.row_max)
        self.column_position = max(self.column_position, self.column_min)
        self.column_position = min(self.column_position, self.column_max)

        # At random intervals, jump to a random position in the world
        if np.random.random_sample() < self.jump_fraction:
            self.column_position = np.random.random_integers(self.column_min,
                                                             self.column_max)
            self.row_position = np.random.random_integers(self.row_min,
                                                          self.row_max)
        self.row_history.append(self.row_position)
        self.column_history.append(self.column_position)

        # Create the sensory input vector
        fov = self.image_data[int(self.row_position - self.fov_height / 2):
                              int(self.row_position + self.fov_height / 2),
                              int(self.column_position - self.fov_width / 2):
                              int(self.column_position + self.fov_width / 2)]
        center_surround_pixels = wtools.center_surround(fov,
                                                        self.fov_span,
                                                        self.fov_span)
        unsplit_sensors = center_surround_pixels.ravel()
        self.sensors = np.concatenate((np.maximum(unsplit_sensors, 0),
                                       np.abs(np.minimum(unsplit_sensors, 0))))

        self.reward = 0
        rewarded_column = (np.abs(self.column_position - self.target_column) <
                           self.reward_region_width / 2)
        rewarded_row = (np.abs(self.row_position - self.target_row) <
                        self.reward_region_width / 2)
        if rewarded_column and rewarded_row:
            self.reward += self.reward_magnitude

        return self.sensors, self.reward


    def visualize_world(self, brain):
        """
        Show what is going on in BECCA and in the world.
        """
        if self.print_features:
            projections = brain.get_index_projections()[0]
            wtools.print_pixel_array_features(
                projections,
                self.fov_span ** 2 * 2,
                0,
                self.fov_span,
                self.fov_span,
                world_name=self.name)

        # Periodically display the history and inputs as perceived by BECCA

        print(' '.join(["world is", str(self.timestep), "timesteps old."]))
        fig = plt.figure(11)
        plt.clf()
        plt.plot(self.row_history, 'k.')
        plt.title("Row history")
        plt.xlabel('time step')
        plt.ylabel('position (pixels)')
        fig.show()
        fig.canvas.draw()
        fig = plt.figure(12)
        plt.clf()
        plt.plot(self.column_history, 'k.')
        plt.title("Column history")
        plt.xlabel('time step')
        plt.ylabel('position (pixels)')
        fig.show()
        fig.canvas.draw()

        fig = plt.figure(13)
        clip = (self.sensors[:int(len(self.sensors)/2.)] -
                self.sensors[int(len(self.sensors)/2.):] + 1.) / 2.
        sensed_image = np.reshape(clip, (self.fov_span, self.fov_span))
        plt.gray()
        plt.imshow(sensed_image, interpolation='nearest')
        plt.title("Image sensed")
        fig.show()
        fig.canvas.draw()
