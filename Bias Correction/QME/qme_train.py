import numpy as np
import xarray as xr
from qme_utils import *
from qme_vars import *


# def make_dist(var, data):
#     """
#     Calculate histograms at a given resolution (default 501) for a set of input data, divided by month
#     Inputs:
#     data - the data for which histograms will be calculated, assumed to have a "time" dimension
#     var - the variable of the data, used for scaling the data before counting
#     reso - the number of bins. This should not be adjusted unless scale_data, unscale_data and limit_data are also adjusted to reflect the new value
#     Returns:
#     output - the original data converted into histograms (with the "time" dimension replaced by "month" and "values" dimensions)
#     """
#     var = get_qme_var(var)
#     reso = var.bin_count()
#     month_values = data.time.values.astype('datetime64[M]').astype(int) % 12
#     def count_dist(values):
#         dist = np.zeros((12, reso))

#         normalized = var.scale_data(var.limit_data(values))

#         # special rounding function used to correct Numpy rounding towards evens - see comments in qme_utils
#         rounded = round_half_up(normalized)

#         # original version
#         # adjusted = np.round(var.scale_data(var.limit_data(values))).astype(int)
        
#         for i, value in enumerate(rounded):
#             if value in range(reso):
#                 dist[month_values[i]][value] += 1
#         return dist
    
#     output = xr.apply_ufunc(count_dist, data, input_core_dims = [["time"]], output_core_dims = [["month", "values"]], vectorize = True, 
#                             output_dtypes = [np.float32], dask_gufunc_kwargs = {"output_sizes": {"month": 12, "values": reso}}, dask = 'parallelized')
#     output = output.assign_coords({"month": np.arange(12) + 1, "values": np.arange(reso)})
    
#     return output

def make_dist(var, data):
    """
    Calculate histograms at a given resolution (default 501) for a set of input data, divided by month
    Inputs:
    data - the data for which histograms will be calculated, assumed to have a "time" dimension
    var - the variable of the data, used for scaling the data before counting
    reso - the number of bins. This should not be adjusted unless scale_data, unscale_data and limit_data are also adjusted to reflect the new value
    Returns:
    output - the original data converted into histograms (with the "time" dimension replaced by "month" and "values" dimensions)
    """
    var = get_qme_var(var)
    reso = var.bin_count()
    month_values = data.time.values.astype('datetime64[M]').astype(int) % 12
    def count_dist(values):
        dist = np.zeros((12, reso))

        normalized = var.scale_data(var.limit_data(values))

        # special rounding function used to correct Numpy rounding towards evens - see comments in qme_utils
        rounded = round_half_up(normalized)

        # original version
        # adjusted = np.round(var.scale_data(var.limit_data(values))).astype(int)
        
        for i, value in enumerate(rounded):
            if value in range(reso):
                dist[month_values[i]][value] += 1
        return dist
    
    output = xr.apply_ufunc(count_dist, data, input_core_dims = [["time"]], output_core_dims = [["month", "values"]], vectorize = True, 
                            output_dtypes = [np.float32], dask_gufunc_kwargs = {"output_sizes": {"month": 12, "values": reso}}, dask = 'parallelized')
    output = output.assign_coords({"month": np.arange(12) + 1, "values": np.arange(reso)})
    
    return output

def find_means(data, mean_smth = 31, ref_year = 15):
    """
    Calculate a mean value for each year. Can also be smoothed and converted to an anomaly compared to a relative reference year.
    Inputs:
    data - the data used to find the mean values. Must have a 'time' dimension
    mean_smth - how many years to smooth the mean values over
    ref_year - the reference year to subtract from all years' mean value, relative to the first year (year 0) of the provided data. Set to None to skip.
    """
    
    # extract year values from datetime data coordinates (as these are not passed into the vectorized function)
    year_values = data.time.values.astype('datetime64[Y]').astype(int)
    min_year = year_values.min()
    total_years = year_values.max() - min_year + 1
    year_values = year_values - min_year
    
    def calc_means(data_loc):
        offset = 0 # for leap year check
        mean_year = np.zeros((total_years))

        # not enough years to calculate mean
        if total_years < mean_smth:
            return mean_year
        
        # add daily values to each year's total
        for i in range(total_years):
            start_index = i * 365 + offset

            # check for leap year (i.e. exclude 366th day if there is one)
            # just for consistency with IDL, though that was probably out of convenience
            # we could adjust this to specifically include the 366th day too
            if year_values[start_index] != i:
                offset += 1
                start_index += 1
                
            end_index = start_index + 365 # end points not included so don't need to subtract 1
            mean_year[i] = data_loc[start_index:end_index].mean()

        # calculate moving average
        if mean_smth > 1:
            mean_year = smooth(mean_year, mean_smth)

        # convert to anomaly with respect to reference year
        if ref_year is not None:
            mean_year = mean_year - mean_year[ref_year]
        
        return mean_year 

    output = xr.apply_ufunc(calc_means, data, 
                            input_core_dims = [["time"]], 
                            output_core_dims = [["values"]], 
                            vectorize = True, 
                            output_dtypes = [np.float32], 
                            dask_gufunc_kwargs = {"output_sizes": {"values": total_years}}, 
                            dask = 'parallelized')
    output = output.assign_coords({"values": np.arange(total_years)})

    return output
    

def future_means(cur_data, fut_data):
    """
    Use a slightly different method to calculate means for future years, by subtracting their means from current means.
    Inputs:
    cur_data - data corresponding to the current time period
    fut_data - data corresponding to the future time period
    """
    cur_means = find_means(cur_data, 0, None)
    fut_means = find_means(fut_data, 0, None)

    cur_overall_mean = cur_means.mean(dim = "values")
    fut_overall_mean = fut_means.mean(dim = "values")

    modified_mean = fut_overall_mean - cur_overall_mean

    # surely there is a better way of doing this...
    fut_means = xr.apply_ufunc(lambda x, y: y, fut_means, modified_mean, 
                               dask = 'parallelized', 
                               output_dtypes = [np.float32], 
                               vectorize = True)
    return fut_means


def calc_qme(var, dist_mdl, dist_obs, xtr = 3, cal_smth = 21, mn_smth = "", mltp = False, mthd = "", lmt = 1.5, lmt_thresh = 10, ssze_lim = 50, force_zero = False):
    """
    Calculates the QME bias correction factors over the given histogram dimension, returning corrected histograms of equal length. Most of the keyword parameters may be kept at their default values.
    Inputs:
    var - the variable being bias corrected
    dist_mdl - the distribution histograms for the model data
    dist_obs - the distribution histograms for the observational (reference) data
    xtr - how long to calculate tail values
    cal_smth - the number of values to be smoothed with a rolling box
    mn_smth - whether to apply a 3 or 5 month moving sum to the histogram data. 
        Can be set to "" for none, "_3mn" for 3 months, "_5mn" for 5 months. 
        Input distributions must have a dimension named "month".
    mltp - whether to use multiplicative scaling during tail correction instead of additive
    mthd - set to "_quick" for slightly faster quantile matching with not much lost 
    lmt - apply a limit in magnitude change (-1 to disable)
    lmt_thresh - the threshold at which to apply lmt
    ssze_lim - the lower limit of sample size to pass the quality check
    force_zero - whether 0 values are retained for certain variables (primarily pr)

    Output:
    A Dataset containing the bias correction values.
    
    """
    var = get_qme_var(var)

    # Include adjacent months if a user option was selected for this.
    if mn_smth == '_3mn':
        dist_mdl = three_mnth_sum(dist_mdl, "month") # Apply a 3-month moving sum to the histogram data.
        dist_obs = three_mnth_sum(dist_obs, "month") # Apply a 3-month moving sum to the histogram data.
        xtr = xtr * 3

    elif mn_smth == '_5mn':
        dist_mdl = three_mnth_sum(dist_mdl, "month") # Apply a 3-month moving sum to the histogram data.
        dist_obs = three_mnth_sum(dist_obs, "month") # Apply a 3-month moving sum to the histogram data.
        dist_mdl = three_mnth_sum(dist_mdl, "month") # Apply another 3-month moving sum to the histogram data, resulting in a [1, 2, 3, 2, 1] type of smoothing centred on a given month.
        dist_obs = three_mnth_sum(dist_obs, "month") # Apply another 3-month moving sum to the histogram data, resulting in a [1, 2, 3, 2, 1] type of smoothing centred on a given month.
        xtr = xtr * 9
        
    mdl_reso = dist_mdl["values"].size
    obs_reso = dist_obs["values"].size

    if mdl_reso != obs_reso:
        raise ValueError(f'Error - mismatched bin counts between between mdl and obs distributions: {mdl_reso}, {obs_reso}')

    reso = mdl_reso
    
    def qcheck(dist_mdl_loc, dist_obs_loc):
    # Do some data quality checks here, noting that users can modify this or add other checks if need be.
        data_ok = 0 # Used to check if data are suitable or not.
        
        dvrs_count = (np.asarray(dist_mdl_loc) > 0).sum() # Check for diversity of data values.
        if dvrs_count <= 1: # Users could also set a different value here for a limit (e.g., <= 5, etc.).
            data_ok -= 1 # Not enough unique input data values.
            
        dvrs_count = (np.asarray(dist_obs_loc) > 0).sum() # Check for diversity of data values.
        if dvrs_count <= 1: # Users could also set a different value here for a limit (e.g., <= 5, etc.).
            data_ok -= 2 # Not enough unique reference data values.

        if min(sum(dist_obs_loc), sum(dist_mdl_loc)) < ssze_lim:
            data_ok -= 4 # Sample size too small.
            
        return data_ok

    
    def factors(dist_mdl_loc, dist_obs_loc):
        reso = len(dist_mdl_loc)
            
        biascorr_cell = np.zeros((reso)) # Array for the calibration values.

        q_val = qcheck(dist_mdl_loc, dist_obs_loc)
        
        # Data unsuitable for bias correction, return blank array instead
        if q_val != 0:
            return biascorr_cell

        scaled_reso = np.arange(reso) # Make an array of integers increasing from 0 to reso.
        unscaled_reso = var.unscale_data(scaled_reso) # This array contains unscaled values, converted from scaled_reso, for use below in calibrating histogram tails.
        
        tots_obs = sum(dist_obs_loc)
        tots_mdl = sum(dist_mdl_loc)
        if tots_obs != tots_mdl: # Scale up histogram with smaller counts so that it has the same sample size as the other histogram.
            if tots_mdl > tots_obs: 
                dist_obs_loc = dist_obs_loc * (tots_mdl/tots_obs)
            else:
                dist_mdl_loc = dist_mdl_loc * (tots_obs/tots_mdl)
        
        # Find the array positions that have data counts > 0.
        vals_obs = np.nonzero(dist_obs_loc > 0)[0]
        vals_mdl = np.nonzero(dist_mdl_loc > 0)[0]
        
        if (len(vals_obs) <= 0) or (len(vals_mdl) <= 0):
            raise ValueError(f'Error - lack of positive values in histogram: {len(vals_obs)}, {len(vals_mdl)}')

        # Calculate cumulative sums forwards and backwards for use later
        # This is to save them from being recalculated every loop
        obs_fwd_sum = np.cumsum(dist_obs_loc)
        mdl_fwd_sum = np.cumsum(dist_mdl_loc)
        obs_bwd_sum = np.flip(np.cumsum(np.flip(dist_obs_loc)))
        mdl_bwd_sum = np.flip(np.cumsum(np.flip(dist_mdl_loc)))
        
        # Do quantile matching over the histogram range.
        if mthd != '_detail': # This is a relatively quick method, with a more detailed option as follows below.
            obs_pos = vals_obs[0] # Position in array for observations data.
            for mdl_pos in range(vals_mdl[0], vals_mdl[-1]): # Position in array for model data.
                while obs_fwd_sum[obs_pos] < mdl_fwd_sum[mdl_pos] and obs_pos < reso - 1:
                    obs_pos += 1
                biascorr_cell[mdl_pos] = obs_pos
                
            if mthd != '_quick': # Can skip this for some extra speed, with relatively little reduction in quality.
                obs_pos = vals_obs[-1] # Position in array for observations data.
                for mdl_pos in range(vals_mdl[-1], vals_mdl[0] - 1, -1): # Position in array for model data.
                    while obs_bwd_sum[obs_pos] < mdl_bwd_sum[mdl_pos] and obs_pos >= 0:
                        obs_pos -= 1
                    biascorr_cell[mdl_pos] += obs_pos
                    
                # Use the average value from the increasing and decreasing method versions above.
                biascorr_cell /= 2

        else:
            raise NotImplementedError("Detailed matching not implemented in Python translation of QME")
        
        # Now focus on tails.
        bias_tails = np.zeros((reso))
        
        # Lower tail.
        xtr_low = vals_mdl[0] # 0 # Find bias at the xtr lowest mdl value.
        while(mdl_fwd_sum[xtr_low] < xtr):
            xtr_low += 1 # Position in histogram of xtr lowest value.
        xtr_low += 1
        val = biascorr_cell[xtr_low]
        val = var.unscale_data(val)
        if mltp:
            bias = val/unscaled_reso[xtr_low]
            bias_tails[:xtr_low + 1] = unscaled_reso[:xtr_low + 1] * bias
        else:
            bias = val - unscaled_reso[xtr_low] # Bias, as an unscaled anomaly.
            bias_tails[:xtr_low + 1] = unscaled_reso[:xtr_low + 1] + bias
        
        # Upper tail.
        xtr_high = vals_mdl[-1] # Find bias at the xtr highest mdl value.
        while(mdl_bwd_sum[xtr_high] < xtr):
            xtr_high -= 1 # Position in histogram of xtr highest value.
        xtr_high -= 1
        val = biascorr_cell[xtr_high]
        val = var.unscale_data(val)
        if mltp:
            bias = val/unscaled_reso[xtr_high]
            bias_tails[xtr_high:] = unscaled_reso[xtr_high:] * bias
        else:
            bias = val - unscaled_reso[xtr_high] # Bias, as an unscaled anomaly.
            bias_tails[xtr_high:] = unscaled_reso[xtr_high:] + bias
        
        if force_zero: # Ensure data for pr are >= 0. Users can do similar here for other variables if need be.
            bias_tails = np.clip(bias_tails, 0, None)
            bias_tails[0] = 0  # Ensure zero rainfall is retained as zero.
        
        bias_tails = var.scale_data(bias_tails)
        biascorr_cell[:xtr_low + 1] = bias_tails[:xtr_low + 1]
        biascorr_cell[xtr_high:] = bias_tails[xtr_high:]
        
        # Do some post-processing
        biascorr_cell = var.unscale_data(biascorr_cell)
        if lmt != -1: # Apply a limit of lmt (in %) for the magnitude change.
            for x in range(reso):
                if biascorr_cell[x] > lmt_thresh: biascorr_cell[x] = min([biascorr_cell[x], unscaled_reso[x]*lmt])

        if cal_smth > 1:
            biascorr_cell = smooth(biascorr_cell - unscaled_reso, cal_smth) + unscaled_reso # Apply a smoothing to the bias correction values (using anomalies, based on subtracting unscaled_reso).
        
        if force_zero: # Ensure data for pr are >= 0. Users can do similar here for other variables if need be.
            biascorr_cell = np.clip(biascorr_cell, 0, None)
            
        biascorr_cell = var.scale_data(biascorr_cell) - scaled_reso # Scale the array and convert to anomaly for storing in the gridded array

        if force_zero:
            biascorr_cell[0] = 0 # Ensure zero rainfall is retained as zero.
            
        return biascorr_cell

    bc_factors = xr.apply_ufunc(factors, dist_mdl, dist_obs, 
                                input_core_dims = [["values"], ["values"]], 
                                output_core_dims = [["values"]], 
                                vectorize = True, 
                                output_dtypes = [np.float32], 
                                dask = 'parallelized')

    return bc_factors
