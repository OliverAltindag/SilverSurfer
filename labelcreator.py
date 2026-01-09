import pandas as pd
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d

def generate_labels_from_csv(T_final, csv_file_path, label_window_seconds=10.0, sigma_seconds=2.0, **kwargs):
    """
    Generates Gaussian-smoothed labels for magnetic switchback edges.
    Function header adapted to match 'generate_labels_from_csv' for pipeline compatibility.
    
    Parameters:
    -----------
    T_final : array-like
        The time array of your magnetic field data (datetime64[ns]).
    csv_file_path : str
        Path to the catalog CSV containing 'spike Start Time' and 'spike End Time'.
    label_window_seconds : float
        The duration (in seconds) of the "core" label (the '1' region).
        Replaces the old 'edge_width_indices' (which was approx 45 steps).
    sigma_seconds : float
        The standard deviation of the Gaussian smoothing kernel (in seconds).
    **kwargs : dict
        Absorbs legacy arguments like 'edge_width_indices' to prevent crashes.
    
    Returns:
    --------
    labels : np.array
        A (N, 1) float32 array with soft labels (values between 0 and ~1).
        
    Short events: Labeled entirely as 1.
    Long events: Only the Start and End regions are labeled 1. 
    The "Steady State" middle is left as 0 to match Ricker/Haar wavelet blindness.
    This is how the model was made, bc of the bracketing of the Haar and multiplication gate.
    
    IoU will be straight cheeks, but stuff will be detected.
    """
    
    # just check if its even there
    if not os.path.exists(csv_file_path):
        print(f"Error: Label file {csv_file_path} not found.")
        return np.zeros((len(T_final), 1), dtype='float32')

    # data load
    df_labels = pd.read_csv(csv_file_path, sep=',') 
    df_labels.columns = df_labels.columns.str.strip()

    # nsure T_final is proper datetime64 format
    if not np.issubdtype(T_final.dtype, np.datetime64):
        T_final = pd.to_datetime(T_final).to_numpy()
    
    # this makes the function resolution-independent
    # we take the median diff to ignore potential gaps
    dt_nanos = np.median(np.diff(T_final)).astype('float64')
    dt_seconds = dt_nanos * 1e-9
    
    if dt_seconds <= 0:
        print("Error: Invalid time steps (dt <= 0). Check T_final for duplicates.")
        return np.zeros((len(T_final), 1), dtype='float32')

    # seconds to array indices
    width_indices = int(np.ceil(label_window_seconds / dt_seconds))
    sigma_indices = sigma_seconds / dt_seconds
    
    # this will be identical bc of the universal clock
    print(f"Sampling Rate (dt): {dt_seconds:.4f} s")
    print(f"Core Label Width:   {width_indices} pixels ({label_window_seconds}s)")
    print(f"Gaussian Sigma:     {sigma_indices:.2f} pixels ({sigma_seconds}s)")

    # binary mask
    binary_labels = np.zeros((len(T_final), 1), dtype='float32')

    # try to parse the times
    try:
        # strip string whitespace before parsing
        start_strs = df_labels['spike Start Time'].astype(str).str.strip()
        end_strs = df_labels['spike End Time'].astype(str).str.strip()
        
        # Parse using 'mixed' format to handle variations
        start_times = pd.to_datetime(start_strs, format='mixed')
        end_times = pd.to_datetime(end_strs, format='mixed')
    except Exception as e:
        print(f"Date Parsing Error: {e}")
        return binary_labels

    # labeling Loop (Binary Phase)
    # iterate through every event in the CSV
    for start_t, end_t in zip(start_times, end_times):
        # explicitly convert Pandas Timestamp to Numpy datetime64
        # this was a pain in the ass to figure out
        st_np = np.datetime64(start_t)
        et_np = np.datetime64(end_t)
        # Find array indices for start and end
        idx_start = np.searchsorted(T_final, st_np, side='left')
        idx_end = np.searchsorted(T_final, et_np, side='right')
        
        # Keep indices inside array bounds
        idx_start = max(0, idx_start)
        idx_end = min(len(T_final), idx_end)

        # Only proceed if we have valid duration
        if idx_end > idx_start:
            # Edge-Only Labeling
            # label entry and exit edges regardless of event duration
            
            # sTart Edge (Entry)
            # clamp end of the start label to be at most idx_end to avoid spilling over
            s_end = min(idx_start + width_indices, idx_end)
            binary_labels[idx_start:s_end] = 1.0
            
            # End Edge (Exit) 
            # Clamp start of the end label to be at least idx_start to avoid spilling back
            e_start = max(idx_end - width_indices, idx_start)
            binary_labels[e_start:idx_end] = 1.0
            
            # middle remains 0.0 (the blind spot)
            # the gate makes it so i kinda have to do this but i also just straight up guess
            # iou metric becomes useless
            # will fix this at some point
            # or not

    # smoothing
    # axis=0 smooths along the time dimension
    # truncate=4.0 cuts the filter off at 4 sigmas (improves performance)
    if sigma_indices > 0:
        smooth_labels = gaussian_filter1d(binary_labels, sigma=sigma_indices, axis=0, mode='constant', cval=0.0, truncate=4.0)
        
        # Optional: Normalize peak to 1.0? 
        # Usually for soft labels, you just want the shape. 
        # But if you want max probability to be 1, you can clip or scale.
        # Here we just clip to ensure no floating point weirdness > 1
        smooth_labels = np.clip(smooth_labels, 0, 1.0)
        return smooth_labels
    
    return binary_labels
