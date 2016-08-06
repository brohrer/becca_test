"""
One-dimensional visual servo task.

This task gives Becca a chance to build a comparatively large number
of sensors into a few informative features. However, due to the
construction of the task, it's not strictly necessary to build
complex features to do well on it.
"""
from __future__ import print_function
import os

import matplotlib.pyplot as plt
import numpy as np

import becca.connector
from becca_test.base_world import World as BaseWorld
import becca_test.world_tools as wtools


class World(BaseWorld):
    """
    One-dimensional visual servo world

    In this world, Becca can direct its gaze left and right
    along a mural. It is rewarded for directing it near the center.
    Optimal performance is a reward of somewhere around .9 per time step.

    Some of this world's attributes are defined in base_world.py.
    The rest are defined below.
    """
    def __init__(self, lifespan=None):
        """
        Set up the world

        Parameters
        ----------
        lifespan : int
            The number of time steps to continue the world.
        """
        BaseWorld.__init__(self, lifespan)
        self.name_long = 'one dimensional visual world'
        self.name = 'image_1D'
        print("Entering", self.name_long)

        # fov_span : int
        #     The world pixelizes its field of view into a superpixel array
        #     that is fov_span X fov_span.
        self.fov_span = 5
        self.num_sensors = 2 * self.fov_span ** 2
        self.num_actions = 8
        # jump_fraction : float
        #     The fraction of time steps on which the agent jumps to
        #     a random position.
        self.jump_fraction = .1
        # step_cost : float
        #     The punishment per position step taken.
        self.step_cost = .1

        self.world_visualize_period = 1e6
        self.brain_visualize_period = 1e3
        # print_features : bool
        #     If True, plot and save visualizations of each of the features
        #     each time the world is visualized,
        #     rendered so that they represent what they mean in this world.
        self.print_features = False

        # Initialize the image to be used as the environment
        module_path = os.path.dirname(os.path.abspath(__file__))
        # image_filename : str
        #     The file name of the image including the relative path.
        self.image_filename = os.path.join(module_path,
                                           'images',
                                           'bar_test.png')
        # data : array of floats
        #     The image, read in and stored as a 2D numpy array.
        self.data = plt.imread(self.image_filename)
        # Convert it to grayscale if it's in color
        if self.data.shape[2] == 3:
            # Collapse the three RGB matrices into one b/w value matrix
            self.data = np.sum(self.data, axis=2) / 3.0
        # Define the size of the field of view, its range of
        # allowable positions, and its initial position
        image_width = self.data.shape[1]
        # max_step_size : int
        #     The largest step size allowed, in pixels in the original image.
        self.max_step_size = image_width / 2
        # target_column : int
        #     The column index that marks the center of the rewarded region.
        self.target_column = image_width / 2
        # reward_region_width : int
        #     The width of the region, in number of columns, within which
        #     the center of the field of view gets rewarded.
        self.reward_region_width = image_width / 8
        # noise_magnitude : float
        #     A scaling factor that drives how much inaccurate each movement
        #     will be.
        self.noise_magnitude = 0.1

        # column_history : list if ints
        #     A time series of the location (measured in column pixels) of the
        #     center of the brain's field of view.
        self.column_history = []
        # fov_height, fov_width : float
        #     The height and width (number of rows) of the field of view,
        #     in pixels.
        self.fov_height = np.min(self.data.shape)
        self.fov_width = self.fov_height
        # column_min, column_max : int
        #     The low and high bounds on where the field of view
        #     can be centered.
        self.column_min = np.ceil(self.fov_width / 2)
        self.column_max = np.floor(self.data.shape[1] - self.column_min)
        # column_position : int
        #     The current location of the center of the field of view.
        self.column_position = np.random.random_integers(self.column_min,
                                                         self.column_max)
        # block_width : int
        #     The width of each superpixel, in number of columns.
        self.block_width = self.fov_width / (self.fov_span + 2)
        self.sensors = np.zeros(self.num_sensors)
        self.action = np.zeros(self.num_actions)
        self.reward = 0.


    def step(self, action):
        """
        Advance the world by one time step

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

        # Actions 0-3 move the field of view to a higher-numbered row
        # (downward in the image) with varying magnitudes, and
        # actions 4-7 do the opposite.
        raw_col_step = (self.action[0] * self.max_step_size / 2 +
                        self.action[1] * self.max_step_size / 4 +
                        self.action[2] * self.max_step_size / 8 +
                        self.action[3] * self.max_step_size / 16 -
                        self.action[4] * self.max_step_size / 2 -
                        self.action[5] * self.max_step_size / 4 -
                        self.action[6] * self.max_step_size / 8 -
                        self.action[7] * self.max_step_size / 16)
        noise_factor = (
            self.noise_magnitude * np.random.random_sample() * 2.0 -
            self.noise_magnitude * np.random.random_sample() * 2.0 + 1.)
        column_step = int(raw_col_step * noise_factor)
        self.column_position = self.column_position + column_step
        self.column_position = max(self.column_position, self.column_min)
        self.column_position = min(self.column_position, self.column_max)
        self.column_history.append(self.column_position)

        # At random intervals, jump to a random position in the world.
        if np.random.random_sample() < self.jump_fraction:
            self.column_position = np.random.random_integers(self.column_min,
                                                             self.column_max)
        # Create the sensory input vector.
        fov = self.data[:, int(self.column_position - self.fov_width / 2):
                        int(self.column_position + self.fov_width / 2)]
        # Calculate center surround features for the image.
        center_surround_pixels = wtools.center_surround(fov,
                                                        self.fov_span,
                                                        self.fov_span)
        unsplit_sensors = center_surround_pixels.ravel()
        # These can be positive or negative, so split them into two
        # sets of sensors--one for the positive values and one for the
        # negative ones. Then stack them together for one big sensor array.
        self.sensors = np.concatenate((np.maximum(unsplit_sensors, 0),
                                       np.abs(np.minimum(unsplit_sensors, 0))))

        # Calculate the reward.
        self.reward = 0
        if (np.abs(self.column_position - self.target_column) <
                self.reward_region_width / 2.0):
            self.reward += 1.
        self.reward -= (np.abs(column_step) /
                        self.max_step_size * self.step_cost)
        return self.sensors, self.reward


    def visualize_world(self, brain):
        """
        Show what's going on in the world.
        """
        if self.print_features:
            projections = brain.cortex.get_index_projections(to_screen=True)[0]
            wtools.print_pixel_array_features(
                projections,
                self.fov_span ** 2 * 2,
                0,
                self.fov_span, self.fov_span,
                world_name=self.name)

        # Periodically show the state history and inputs as perceived by Becca.
        print(''.join(["world is ", str(self.timestep), " timesteps old"]))
        fig = plt.figure(11)
        plt.clf()
        plt.plot(self.column_history, 'k.', alpha=.2)
        plt.title(''.join(['Column history for ', self.name]))
        plt.xlabel('time step')
        plt.ylabel('position (pixels)')
        fig.show()
        fig.canvas.draw()

        fig = plt.figure(12)
        sensed_image = np.reshape(
            0.5 * (self.sensors[:len(self.sensors)/2] -
                   self.sensors[len(self.sensors)/2:] + 1),
            (self.fov_span, self.fov_span))
        plt.gray()
        plt.imshow(sensed_image, interpolation='nearest')
        plt.title("Image sensed")
        fig.show()
        fig.canvas.draw()


if __name__ == "__main__":
    becca.connector.run(World())
