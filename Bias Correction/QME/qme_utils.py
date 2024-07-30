import numpy as np
import xarray as xr


class qme_var:
    def __init__(self, lower_lim, upper_lim, max_bin = 500, scaling = 'auto', unscaling = None, verify_override = False):
        self.min = lower_lim
        self.max = upper_lim
        self.reso = max_bin

        if scaling == 'auto':
            if unscaling is not None:
                raise ValueError("Cannot supply unscaling if scaling is not supplied or set to auto")

            self._scaling = lambda x: (self.reso / (self.max - self.min)) * (x - self.min)
            self._unscaling = lambda x: (x / (self.reso / self.max - self.min)) + self.min
            
        else:
            self._scaling = scaling
            if unscaling is None:
                raise ValueError("Unscaling must be supplied if scaling is supplied (it should be the reverse operation)")
            self._unscaling = unscaling
            if not verify_override:
                self.verify_scaling()

    def scale_data(self, data):
        return self._scaling(data)

    def unscale_data(self, data):
        return self._unscaling(data)

    def limit_data(self, data):
        return np.clip(data, self.min, self.max)

    def bin_count(self):
        """
        The total number of bins for this variable, noting that one is added because the end value is included.
        """
        return self.reso + 1

    def verify_scaling(self):
        """
        Verify that the supplied scaling and unscaling functions align with each and the given bin count. 
        """
        scaled_min = self.scale_data(self.min)
        scaled_max = self.scale_data(self.max)

        # -0.5 is the threshold as values between here and 0 can be rounded to 0. Likewise, values between reso and reso + 0.5 can be rounded to reso
        # Checking both sides for min and max in case a negative scaling is involved for some reason
        if scaled_min < -0.5 or scaled_min >= self.reso + 0.5:
            raise ValueError(f'Scaling function produces out of bound value {scaled_min} when applied to min value {self.min} - \n' +
                             f'Ensure scaling function only produces values between 0 and {self.reso} when applied between given limits {self.min} and {self.max}.')
            
        if scaled_max < -0.5 or scaled_max >= self.reso + 0.5:
            raise ValueError(f'Scaling function produces out of bound value {scaled_max} when applied to max value {self.max} - \n' +
                             f'Ensure scaling function only produces values between 0 and {self.reso} when applied between given limits {self.min} and {self.max}.\n' +
                             f'Alternatively, supply a higher bin count.')
            
        # Check that scaled values unscale back to their original value
        if not np.isclose(self.min, self.unscale_data(scaled_min)):
            raise ValueError(f'Failed to symmetrically unscale min value {self.min} after scaling - check the unscaling function.')
            
        if not np.isclose(self.max, self.unscale_data(scaled_max)):
            raise ValueError(f'Failed to symmetrically unscale max value {self.max} after scaling - check the unscaling function.')


def round_half_up(data):
    """
    Round .5 values up instead of towards even (the behaviour Numpy uses) for consistency with IDL
    """

    # Numpy rounds towards evens (i.e. 1.5 and 2.5 will both round to 2, instead of 2 and 3 respectively).
    # To correct for this, we compare the rounded result to the rounded result of the original array plus one:
    # if the result is 2 then the original was rounded down instead of up.
    # These cases are isolated with the division and floor operations (so the other results, 0 and 1, will all become 0)
    # and added to the original rounding result before being converted to integers
    rounded = np.round(data)
    correction = np.floor((np.round(data + 1) - rounded) / 2)
    adjusted = (rounded + correction).astype(int)
    return adjusted


def three_mnth_sum(data, dim = "month"):
    """
    Procedure to make 3-month moving sum, with rolling around the edges.
    Inputs: 
    data - a DataArray with the specified dimension of size 12 (representing months)
    dim (optional) - specify dimension name in case it is not called 'month'
    """

    if not(data[dim].size == 12):
        raise ValueError(f'Month dimension must be of size 12, given array had size {data[dim].size}')

    # calculate rolling sum
    summed_dat = data.rolling({dim: 3}, center = True).sum()

    # manually calculate Jan + Dec since rolling does not work on edges
    summed_dat[{dim: 0}] = data[{dim: 0}] + data[{dim: 1}] + data[{dim: 11}]
    summed_dat[{dim: 11}] = data[{dim: 0}] + data[{dim: 10}] + data[{dim: 11}]
        
    return summed_dat
    

# written to replicate IDL smoothing
def smooth(data, width):
    """
    Smoothes data using a moving box of a given width. Near edges, the border value is repeated to fill the rest of the box
    Inputs: 
    data - the data being smoothed, assumed to be a one-dimensional array. If not, the smoothing will occur across the first dimension
    width - the width of the moving box. If even, it is converted into an odd number by adding 1
    Returns:
    smoothed - the original data with smoothing applied
    """
    # IDL will add 1 to any even number given as an argument
    if width % 2 == 0:
        width += 1

    # the IDL function was called with "/EDGE_TRUNCATE", meaning that along the edges out of bound values were just filled with the edge value
    # instead of being replaced by NaNs
    side = width // 2
    box = side * data[0] + sum(data[:side + 1])

    smoothed = data.copy()
    
    for i in range(data.size):
        smoothed[i] = box/width

        # modify the box by removing the left-most value in the box and adding the next one to the right
        # the checks are for the edge cases to preserve "/EDGE_TRUNCATE" functionality
        box -= data[max(i - side, 0)]
        box += data[min(i + side + 1, data.size - 1)]
        
    return smoothed
