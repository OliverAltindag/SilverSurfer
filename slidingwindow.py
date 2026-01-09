import numpy as np

def create_sliding_windows(
    ricker_br,       # Shape: (n_scales, full_time_length)
    ricker_vr,       # Shape: (n_scales, full_time_length)
    haar_br,         # Shape: (full_time_length,)
    labels,          # Shape: (full_time_length, 1) or None
    window_size=256, # UPDATED: Expanded to 256 for Encounter 5 events
    stride=64        # UPDATED: Increased stride to maintain 75% overlap
):
    """
    Slices continuous time-series tensors into overlapping windows for the CNN.
    """
    # setup
    time_len = haar_br.shape[0]
    n_scales = ricker_br.shape[0]
    
    if time_len < window_size:
        print(f"Error: Data length {time_len} is shorter than window {window_size}")
        return None, None, None, None, None

    # calculate number of windows
    # integer division ignores the trailing data that doesn't fit a full window
    # sorry!
    num_windows = (time_len - window_size) // stride + 1
    
    print(f"Slicing {time_len} time steps into {num_windows} windows (Stride={stride})...")

    # pre-allocate Memory
    # we allocate standard float32 arrays
    # we is me
    # dimensions: (Batch_Size, Height/Scales, Width/Time, Channels)
    
    # Ricker: (Batch, 64, Window, 1)
    batch_r_br = np.zeros((num_windows, n_scales, window_size, 1), dtype='float32')
    batch_r_vr = np.zeros((num_windows, n_scales, window_size, 1), dtype='float32')
    
    # Haar: (Batch, Window, 1)
    batch_h_br = np.zeros((num_windows, window_size, 1), dtype='float32')
    
    # Local Labels: (Batch, Window, 1)
    batch_lbls = np.zeros((num_windows, window_size, 1), dtype='float32')
    
    # the sliding Loop
    idx = 0
    for i in range(0, time_len - window_size + 1, stride):
        # Define the window boundaries
        start = i
        end = i + window_size
        
        batch_r_br[idx, :, :, 0] = ricker_br[:, start:end]
        batch_r_vr[idx, :, :, 0] = ricker_vr[:, start:end]
        batch_h_br[idx, :, 0] = haar_br[start:end]
        
        if labels is not None:
            batch_lbls[idx, :, 0] = labels[start:end, 0]
        idx += 1
        
    # generate global labels
    # Logic: If the local mask has a switchback (1) anywhere in the window,
    # the global label for that window is 1
    # lowkey didnt work terribly
    if labels is not None:
        Y_global = np.any(batch_lbls > 0.5, axis=(1, 2)).astype('float32').reshape(-1, 1)
    else:
        Y_global = None

    return batch_r_br, batch_r_vr, batch_h_br, batch_lbls, Y_global
