"""
A few functions that are useful to multiple worlds
"""
from __future__ import print_function
import os

import matplotlib.pyplot as plt
import numpy as np

import becca.tools as tools


def center_surround(fov, fov_horz_span, fov_vert_span, verbose=False):
    """
    Convert a 2D array of b/w pixel values to center-surround

    Parameters
    ----------
    fov : 2D array of floats
         Pixel values from the field of view.
    fov_horz_span : int
        Desired number of center-surround superpixel columns.
    fov_vert_span: int
        Desired number of center-surround superpixel rows.
    verbose : bool
        If True, print more information to the console.

    Returns
    -------
    center_surround_pixels :  2D array of floats
        The center surround values corresponding to the inputs.
    """
    fov_height = fov.shape[0]
    fov_width = fov.shape[1]
    block_width = float(fov_width) / float(fov_horz_span + 2)
    block_height = float(fov_height) / float(fov_vert_span + 2)
    super_pixels = np.zeros((fov_vert_span + 2, fov_horz_span + 2))
    center_surround_pixels = np.zeros((fov_vert_span, fov_horz_span))
    # Create the superpixels by averaging pixel blocks
    for row in range(fov_vert_span + 2):
        for col in range(fov_horz_span + 2):
            super_pixels[row][col] = np.mean(
                fov[int(float(row) * block_height):
                    int(float(row + 1) * block_height),
                    int(float(col) * block_width) :
                    int(float(col + 1) * block_width)])
    for row in range(fov_vert_span):
        for col in range(fov_horz_span):
            # Calculate a center-surround value that represents
            # the difference between the pixel and its surroundings.
            # Weight the N, S, E, and W pixels by 1/6 and
            # the NW, NE, SW, and SE pixels by 1/12, and
            # subtract from the center.
            center_surround_pixels[row][col] = (
                super_pixels[row + 1][col + 1] -
                super_pixels[row][col + 1] / 6 -
                super_pixels[row + 2][col + 1] / 6 -
                super_pixels[row + 1][col] / 6 -
                super_pixels[row + 1][col + 2] / 6 -
                super_pixels[row][col] / 12 -
                super_pixels[row + 2][col] / 12 -
                super_pixels[row][col + 2] / 12 -
                super_pixels[row + 2][col + 2] / 12)
    if verbose:
        # Display the field of view clipped from the original image
        plt.figure("fov")
        plt.gray()
        img = plt.imshow(fov)
        img.set_interpolation('nearest')
        plt.title("field of view")
        plt.draw()

        # Display the pixelized version, a.k.a. superpixels
        plt.figure("super_pixels")
        plt.gray()
        img = plt.imshow(super_pixels)
        img.set_interpolation('nearest')
        plt.title("super pixels")
        plt.draw()

        # Display the center-surround filtered superpixels
        plt.figure("center_surround_pixels")
        plt.gray()
        img = plt.imshow(center_surround_pixels)
        img.set_interpolation('nearest')
        plt.title("center surround pixels")
        plt.draw()
    return center_surround_pixels


def print_pixel_array_features(projections,
                               num_pixels_x2,
                               start_index,
                               fov_horz_span,
                               fov_vert_span,
                               directory='log',
                               world_name='',
                               interp='nearest'):
    """
    Interpret an array of center-surround pixels as an image.

    Parameters
    ----------
    directory : str
       The directory into which the feature images will be saved.
       Default is 'log'.
    fov_horz_span, fov_vert_span : int
        The number of pixels in the horizontal (columns) and vertical (rows)
        directions.
    interp : str
        The method of interpolation that matplotlib will use when creating
        the image. Default is 'nearest'.
    num_pixels_x2 : int
        Twice the number of center-surround superpixels.
    projections : list of list of array of floats
        This is the set of all the projections of all the features from
        all the ``ZipTie``s onto sensors.
    start_index : int
        Which index in the projection arrays marks the beginning of
        the center-surround sensors.
    world_name : str
        A base name for the image filenames, associated with the world.
    """
    num_levels = len(projections)
    for level_index in range(num_levels):
        for feature_index in range(len(projections[level_index])):
            plt.close(99)
            feature_fig = plt.figure(num=99)

            # Get the pixel array for the projection image.
            projection_image = (visualize_pixel_array_feature(
                projections[level_index][feature_index]
                [start_index:start_index + num_pixels_x2],
                fov_horz_span, fov_vert_span, array_only=True))
            rect = (0., 0., 1., 1.)
            axes = feature_fig.add_axes(rect)
            plt.gray()
            axes.imshow(projection_image,
                        interpolation=interp, vmin=0., vmax=1.)

            # Create a plot of individual features.
            filename = '_'.join(['level', str(level_index).zfill(2),
                                 'feature', str(feature_index).zfill(4),
                                 world_name, 'image.png'])
            full_filename = os.path.join(directory, filename)
            plt.title(filename)
            plt.savefig(full_filename, format='png')


def visualize_pixel_array_feature(sensors,
                                  fov_horz_span=None,
                                  fov_vert_span=None,
                                  level_index=-1,
                                  feature_index=-1,
                                  world_name=None,
                                  save_png=False,
                                  filename='log/feature',
                                  array_only=False):
    """
    Show a visual approximation of an array of center-surround sensorss.

    Parameters
    ----------
    array_only : bool
        If True, calculate but do not plot the pixel values of the image.
        Default is False.
    feature_index : int
        The index of the feature in its ``ZipTie``.
    filename : str
        The base filename under which each feature visualization image
        is saved. The default is 'log/feature'.
    fov_horz_span : int
        Desired number of center-surround superpixel columns.
    fov_vert_span: int
        Desired number of center-surround superpixel rows.
    level_index : int
        The index of the ``ZipTie`` level from which the feature is taken.
    save_png : bool
        If True, save a copy of the visualization as a png. Default is False.
    sensors : array of floats
        This assumes that sensors are arranged as a set of flattened
        superpixel brightness sernsors concatenated with the complementary
        set of flattened superpixel darkness sensors (1. - brightness).
    world_name : str
        The name of the world that generated the features.

    Returns
    -------
    feature_pixels : 2D array of floats
        If ``array_only``, return the array of image values.
    """
    # Calculate the number of pixels that span the field of view
    if fov_horz_span is None:
        n_pixels = sensors.shape[0] / 2.
        fov_horz_span = np.sqrt(n_pixels)
        fov_vert_span = np.sqrt(n_pixels)
    else:
        n_pixels = fov_horz_span * fov_vert_span

    # Maximize contrast
    sensors *= 1. / (np.max(sensors) + tools.epsilon)

    # Calculate the visualization image pixel values.
    pixel_values = ((sensors[0:n_pixels] -
                     sensors[n_pixels:2 * n_pixels]) + 1.0) / 2.0
    feature_pixels = pixel_values.reshape(fov_vert_span, fov_horz_span)

    if array_only:
        return feature_pixels
    else:
    # Initialize the and plot the figure.
        level_str = str(level_index).zfill(2)
        feature_str = str(feature_index).zfill(3)
        fig_title = ' '.join(('Level', level_str, 'Feature', feature_str,
                              'from', world_name))
        fig_name = ' '.join(('Features from ', world_name))
        fig = plt.figure(tools.str_to_int(fig_name))
        fig.clf()
        plt.gray()
        fig.add_axes(0., 0., 1., 1., frame_on=False)
        plt.imshow(feature_pixels, vmin=0.0, vmax=1.0,
                   interpolation='nearest')
        plt.title(fig_title)

        # Save the image file.
        if save_png:
            filename = ''.join([filename, '_', world_name, '_',
                                level_str, '_', feature_str, '.png'])
            fig.savefig(filename, format='png')

        # Force an update of the screen display.
        fig.show()
        fig.canvas.draw()


def resample2D(array, num_rows, num_cols):
    """
    Resample a 2D array to get one that has num_rows and num_cols.

    Use and approximate nearest neighbor method to resample to the
    pixel on the next lower row and column..

    Parameters
    ----------
    array : 2D array of floats
        The array to resample from.
    num_cols, num_rows : ints
        The number of rows and columns to include in the
        evenly-spaced grid resampling.

    Returns
    -------
    resampled_array : 2D array of floats
        The resampled version of the array with the appropriate dimensions.
    """
    rows = (np.linspace(0., .9999999, num_rows) *
            array.shape[0]).astype(np.int)
    cols = (np.linspace(0., .9999999, num_cols) *
            array.shape[1]).astype(np.int)

    if len(array.shape) == 2:
        resampled_array = array[rows, :]
        resampled_array = resampled_array[:, cols]
    if len(array.shape) == 3:
        resampled_array = array[rows, :, :]
        resampled_array = resampled_array[:, cols, :]
    return resampled_array
