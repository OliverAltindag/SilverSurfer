import numpy as np
import pandas as pd
import glob
import os
from datetime import datetime

def apply_loose_fit(data_array, window_size=5):
    """
    Cleans velocity data using a Median Filter to ignore outliers (rogue dots)
    and Linear Interpolation to bridge gaps.
    """
    series = pd.Series(data_array)
    
    # filter instrument errors (-1e31)
    # which is a manually added erorr when data is missing
    series[series < -10000] = np.nan
    series[series > -10000] = np.nan
    
    # median filtering
    # min_periods=3 ensures that isolated dots
    # are removed because they don't follow the true pattern 
    # and are a matter of equipment recalibration
    v_median = series.rolling(window=window_size, center=True, min_periods=3).median()
    
    # bridge the gap, simple linear fit
    v_bridged = v_median.interpolate(method='linear', limit_direction='both')
    
    # smoothing
    # Prevents noise in the Haar transform by softening corners
    v_smooth = v_bridged.rolling(window=window_size, center=True, min_periods=1).mean()
    
    # return numpy array, filling edges with 0 if necessary
    return v_smooth.fillna(0).values

def get_haar_features(data):
    """
    Performs manual Haar MRA decomposition to find high frequency edges.
    Returns the high frequency signal only, which will show a spike if present.
    """
    n = len(data)
    
    # handle odd lengths temporarily
    if n % 2 != 0: 
        proc_data = data[:-1]
    else:
        proc_data = data
        
    # fine Scale
    f_fine = proc_data
    
    # coarse Scale: j-1
    pairs = f_fine.reshape(-1, 2)
    averages = pairs.mean(axis=1)
    f_coarse = np.repeat(averages, 2)
    
    # bracketing appearance
    details = f_fine - f_coarse
    
    # restore length if truncated
    if n % 2 != 0:
        details = np.append(details, 0)
        
    return details

def _ricker_wavelet(points, a):
    """
    Generates the Mexican Hat shape to fit into the data
    Similar to seismic data fitting
    We will slide a window over the series data which should enable the 
    wavelet to find the switchbacks peak
    """
    # just creates the shape using math
    A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    xsq = vec**2
    mod = (1 - xsq / wsq)
    gauss = np.exp(-xsq / (2 * wsq))
  
    return A * mod * gauss

def get_ricker_features(data, scales=np.arange(1, 64)):
    """
    Performs Continuous Wavelet Transform (CWT) using Ricker.
    Returns a 2D Spectrogram (Scales x Time).
    """
    output = np.zeros((len(scales), len(data)))
    
    for idx, width in enumerate(scales):
        # create wavelet kernel for this scale
        # width * 10 ensures we capture the tails of the hat
        # if the ricker is too large, its area won't be zero in the window
        # we will get artifacts in the corner which will not affect 
        # training but will look rather stupid
        # this is one of the limitations of using the ricker in this sliding method
        # the CNN will have to learn to ignore this with training
        # which is feasible as it is supervised
        num_points = min(10 * width, len(data))
        wavelet_data = _ricker_wavelet(num_points, width)
        
        # convolve (mode='same' keeps output length equal to input)
        output[idx, :] = np.convolve(data, wavelet_data, mode='same')
        
    return np.abs(output)
