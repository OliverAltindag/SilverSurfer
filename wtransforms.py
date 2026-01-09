import numpy as np
import pandas as pd
import glob
import os
from scipy.signal import fftconvolve

def apply_loose_fit(data_array, window_size=7):
    """
    Cleans data using a Median Filter to ignore outliers (rogue dots)
    and Linear Interpolation to bridge gaps.
    
    UPDATED: Default window_size increased to 7 to handle clumps of error values
    often seen during violent switchback events.
    """
    series = pd.Series(data_array)
    
    # filter instrument errors (-1e31)
    # which is a manually added erorr when data is missing
    # you will see i do this elsewhere too, bc inintially i wasnt applying this to the B_R data
    series[series < -10000] = np.nan
    series[series > 10000] = np.nan
    
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

    # such pretty and short code
    return details

def _ricker_wavelet(points, a):
    """
    Generates the Mexican Hat shape to fit into the data
    Similar to seismic data fitting
    We will slide a window over the series data which should enable the 
    wavelet to find the switchbacks peak

    insallah
    """
    # just creates the shape using math
    # dont really know what to say other than that
    # enjoy?
    A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    xsq = vec**2
    mod = (1 - xsq / wsq)
    gauss = np.exp(-xsq / (2 * wsq))
    return A * mod * gauss
    
def get_ricker_features_fast(data, scales=np.arange(1, 65)):
    """
    FFT-based Continuous Wavelet Transform (CWT) using Ricker.
    """
    n_points = len(data)
    n_scales = len(scales)
    
    # pre-allocate output
    output = np.zeros((n_scales, n_points))
    
    # use fftconvolve
    for idx, width in enumerate(scales):
        # create kernel
        # ensure kernel is odd length for perfect centering
        len_wavelet = min(10 * width, n_points)
        if len_wavelet % 2 == 0: len_wavelet += 1
            
        A = 2 / (np.sqrt(3 * width) * (np.pi**0.25))
        wsq = width**2
        vec = np.arange(0, len_wavelet) - (len_wavelet - 1.0) / 2
        xsq = vec**2
        mod = (1 - xsq / wsq)
        gauss = np.exp(-xsq / (2 * wsq))
        wavelet = A * mod * gauss
        
        # FFT convolve is significantly faster for large windows
        # mode='same' handles the padding/centering
        # saved my wee little surface from exploding
        output[idx, :] = fftconvolve(data, wavelet, mode='same')
        
    return output
