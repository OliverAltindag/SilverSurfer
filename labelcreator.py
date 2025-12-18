import pandas as pd
import numpy as np
import os

def generate_labels_from_csv(T_final, csv_file_path, edge_width_indices=45):
    """
    Parses catalog CSV and maps events to the T_final array indices.
    Produces the binary target vector (Y) for CNN training.
    
    Short events: Labeled entirely as 1.
    Long events: Only the Start and End regions are labeled 1. 
    The "Steady State" middle is left as 0 to match Ricker/Haar wavelet blindness.
    This is how the model was made, bc of the bracketing.
      
    edge_width_indices: Number of steps to label as '1' from the edge inward.
                        Set to approx 1/3 of WINDOW_SIZE
    """
    # just check if its even there
    if not os.path.exists(csv_file_path):
        print(f"Error: Label file {csv_file_path} not found.")
        return np.zeros((len(T_final), 1), dtype='float32')

    # read the labels
    df_labels = pd.read_csv(csv_file_path)
    df_labels.columns = df_labels.columns.str.strip()
    
    # try to parse the times
    fmt = "%Y-%m-%d/%H:%M:%S"
    try:
        start_times = pd.to_datetime(df_labels['spike Start Time'], format=fmt)
        end_times = pd.to_datetime(df_labels['spike End Time'], format=fmt)
    except ValueError:
        print("Warning: Strict date format failed. Attempting flexible parsing...")
        start_times = pd.to_datetime(df_labels['spike Start Time'])
        end_times = pd.to_datetime(df_labels['spike End Time'])
    
    # label array
    labels = np.zeros((len(T_final), 1), dtype='float32')

    # this prevents the "int vs Timestamp" error by forcing numpy datetime format
    if not np.issubdtype(T_final.dtype, np.datetime64):
        T_final = pd.to_datetime(T_final).to_numpy()

    # ensure T_final is sorted
    if not np.all(T_final[:-1] <= T_final[1:]):
        T_final = np.sort(T_final)

    for start_t, end_t in zip(start_times, end_times):
        # explicitly convert Pandas Timestamp to Numpy datetime64
        # this was a pain in the ass to figure out
        st_np = np.datetime64(start_t)
        et_np = np.datetime64(end_t)

        idx_start = np.searchsorted(T_final, st_np, side='left')
        idx_end = np.searchsorted(T_final, et_np, side='right')
        
        idx_start = max(0, idx_start)
        idx_end = min(len(T_final), idx_end)
        
        if idx_end > idx_start:
            # check duration of the event in indices
            duration = idx_end - idx_start
            
            # 30% of the event duration
            # but cap it at the fixed 'edge_width_indices'
            # to prevent labeling the "blind middle" of massive events
            # this is dynamic
            dynamic_width = int(duration * 0.30)
            use_width = min(dynamic_width, edge_width_indices)
            
            # ensure at least 1 pixel is labeled
            use_width = max(1, use_width)
            
            # if the event is shorter than 2x the edge width, we can't have a "middle hole"
            # so just label the whole thing (it's a short spike/structure)
            if duration <= (2 * use_width):
                labels[idx_start:idx_end] = 1.0
            else:
                # label only the edges
                # start edge
                labels[idx_start : idx_start + use_width] = 1.0
                # end edge
                labels[idx_end - use_width : idx_end] = 1.0
                # middle remains 0.0 (the blind spot)
                # the gate makes it so i kinda have to do this but i also just straight up guess
                # iou metric becomes useless
                # will fix this at some point
                # or not
            
    return labels
