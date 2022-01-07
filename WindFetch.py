import math
import numpy as np
from scipy.ndimage.interpolation import rotate


class WaterBody:
    """
    The Waterbody class
    Requires raster with water classification, resolution and water_id value for water cells
    """

    def __init__(self, landwater, water_id, resolution):
        self.water_id = water_id
        self.landwater = np.where(landwater == self.water_id, -1, np.nan)
        self.resolution = resolution

        # we calculate a rough estimate of padding so we can rotate the image
        # while still maintaining spatial awareness once we rotate it back and
        # remove the same padding we added before rotation
        nrow, ncol = self.landwater.shape
        xlen = self.resolution * ncol
        ylen = self.resolution * nrow
        padwidth = math.ceil(np.sqrt(xlen ** 2 + ylen ** 2) - min([xlen, ylen]))
        self.estimated_pad = int(padwidth)

    @staticmethod
    def padding(array, pad_width, fill_value, inverse=False):
        if inverse == False:
            arr = np.pad(
                array,
                pad_width=pad_width,
                mode="constant",
                constant_values=fill_value,
            )
        else:
            arr = array[pad_width:-pad_width, pad_width:-pad_width]

        return arr

    @staticmethod
    def _fetch_length_vect(array, resolution):
        """Calculate fetch for a given array along axis 1"""
        w = array * -1  # now its nans or 1 for water
        v = w.flatten(order="F")  # flattened by column
        n = np.isnan(v)  # is nan mask
        a = ~n  # not nan mask
        c = np.cumsum(
            a
        )  # add up in sequence for water pixels in a row. More pixels in a row, more fetch
        b = np.concatenate(([0.0], c[n]))  # so diff works with first pixel
        d = np.diff(
            b
        )  # make fetch count from cumsum only increment for a single vector in the plane
        v[n] = -d
        x = np.cumsum(v)
        y = np.reshape(x, w.shape, order="F") * w * resolution
        return y

    def _fetch_single_dir(self, angle):
        # Prepare array for fetch calculation i.e padding and rotating
        array_pad = self.padding(self.landwater, self.estimated_pad, np.nan)
        array_rot = rotate(
            array_pad,
            angle=angle,
            reshape=False,
            mode="constant",
            cval=np.nan,
            order=0,
        )
        array_fetch = self._fetch_length_vect(array_rot, self.resolution)
        array_inv_rot = rotate(
            array_fetch,
            angle=360 - angle,
            reshape=False,
            mode="constant",
            cval=np.nan,
            order=0,
        )
        array_inv_pad = self.padding(
            array_inv_rot, self.estimated_pad, -self.resolution, inverse=True
        )

        return array_inv_pad

    # Main function for calculating fetch from several directions
    def fetch(self, directions, minor_directions=None, minor_interval=None):
        """
        Calculates fetch from arbitrary directions supplied as a list.

        Optionally, fetch can be calculate as the mean of N minor_directions
        centered around each direction with distance minor_interval.
        """
        directions = directions if isinstance(directions, list) else [directions]

        def minor_dir_list(directions, minor_interval, minor_directions):
            minor_seq = [i * minor_interval for i in range(minor_directions)]
            minor_seq_mid = minor_seq[int(len(minor_seq) / 2)]
            all_directions = []
            for d in directions:
                for i in minor_seq:
                    all_directions.append((d + (i - minor_seq_mid)) % 360)

            return all_directions

        def divide_chunks(l, n):
            for i in range(0, len(l), n):
                yield l[i : i + n]

        # Calculate fetch length for each direction
        dir_arrays = []

        if minor_interval and minor_directions:
            all_directions = minor_dir_list(
                directions, minor_interval, minor_directions
            )
            all_dir_arrays = []
            for d in all_directions:
                array_single_dir = self._fetch_single_dir(angle=d)
                all_dir_arrays.append(array_single_dir)

            for i in divide_chunks(all_dir_arrays, minor_directions):
                dir_arrays.append(np.mean(np.stack(i), axis=0))

        else:
            for d in directions:
                array_single_dir = self._fetch_single_dir(angle=d)
                dir_arrays.append(array_single_dir)

        fetch_arrays = np.dstack(dir_arrays)
        return fetch_arrays
