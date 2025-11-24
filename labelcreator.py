import pandas as pd
import numpy as np
import os

def generate_labels_from_csv(T_final, csv_file_path):
    """
    Parses catalog CSV and maps events to the T_final array indices.
    Produces the binary target vector (Y) for CNN training.
    """
    if not os.path.exists(csv_file_path):
        print(f"Error: Label file {csv_file_path} not found.")
        return np.zeros((len(T_final), 1), dtype='float32')

    df_labels = pd.read_csv(csv_file_path)
    df_labels.columns = df_labels.columns.str.strip()
    
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

    # ensure T_final is sorted (required for searchsorted)
    if not np.all(T_final[:-1] <= T_final[1:]):
        T_final = np.sort(T_final)

    for start_t, end_t in zip(start_times, end_times):
        # explicitly convert Pandas Timestamp to Numpy datetime64
        # otherwise it fails
        st_np = np.datetime64(start_t)
        et_np = np.datetime64(end_t)

        idx_start = np.searchsorted(T_final, st_np, side='left')
        idx_end = np.searchsorted(T_final, et_np, side='right')
        
        idx_start = max(0, idx_start)
        idx_end = min(len(T_final), idx_end)
        
        if idx_end > idx_start:
            labels[idx_start:idx_end] = 1.0
            
    return labels
