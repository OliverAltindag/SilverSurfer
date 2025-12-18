import numpy as np
import tensorflow as tf

def prepare_cnn_inputs(
    ricker_br_data,    # 2D: (n_scales, time_length) from Ricker transform
    ricker_vr_data,    # 2D: (n_scales, time_length) from Ricker transform  
    haar_br_data,      # 1D: (time_length,) from Haar high-freq coefficient
    time_length=None
):
    """
    Prepares wavelet-transformed data for CNN input
    Makes the images to put into the CNN
    """
    if time_length is None:
        time_length = len(haar_br_data)

    if time_length % 2 != 0:
        time_length -= 1
    
    # ensure ricker data has correct time dimension
    if ricker_br_data.shape[1] != time_length:
        # truncate or pad if needed
        if ricker_br_data.shape[1] > time_length:
            ricker_br_data = ricker_br_data[:, :time_length]
        else:
            # pad with zeros
            pad_width = time_length - ricker_br_data.shape[1]
            ricker_br_data = np.pad(ricker_br_data, ((0,0), (0, pad_width)), mode='edge')
    # same as above diff data
    if ricker_vr_data.shape[1] != time_length:
        if ricker_vr_data.shape[1] > time_length:
            ricker_vr_data = ricker_vr_data[:, :time_length]
        else:
            pad_width = time_length - ricker_vr_data.shape[1]
            ricker_vr_data = np.pad(ricker_vr_data, ((0,0), (0, pad_width)), mode='edge')

    # here too
    # ensure haar data has correct time dimension
    if len(haar_br_data) != time_length:
        if len(haar_br_data) > time_length:
            haar_br_data = haar_br_data[:time_length]
        else:
            pad_width = time_length - len(haar_br_data)
            haar_br_data = np.pad(haar_br_data, (0, pad_width), mode='edge')
    
    # add channel dimensions for CNN
    ricker_br_input = ricker_br_data.reshape(1, ricker_br_data.shape[0], ricker_br_data.shape[1], 1)
    ricker_vr_input = ricker_vr_data.reshape(1, ricker_vr_data.shape[0], ricker_vr_data.shape[1], 1)
    haar_br_input = haar_br_data.reshape(1, time_length, 1)
    
    return ricker_br_input, ricker_vr_input, haar_br_input

def prepare_batch_inputs(
    ricker_br_batch,    # List of 2D arrays or 3D array (batch, n_scales, time_length)
    ricker_vr_batch,    # List of 2D arrays or 3D array (batch, n_scales, time_length)
    haar_br_batch,      # List of 1D arrays or 2D array (batch, time_length)
    time_length=None
):
    """
    Prepares batch of wavelet-transformed data for CNN input.
    Huzzah!
    """
    # odd todd
    if time_length % 2 != 0:
        time_length -= 1
    
    # same reasoning as the function above
    if time_length is None:
        time_length = len(haar_br_batch[0]) if isinstance(haar_br_batch, list) else haar_br_batch.shape[1]
    batch_size = len(ricker_br_batch) if isinstance(ricker_br_batch, list) else ricker_br_batch.shape[0]
    
    # get shapes from first sample
    n_scales = ricker_br_batch[0].shape[0] if isinstance(ricker_br_batch, list) else ricker_br_batch.shape[1]
    
    # initialize batch arrays
    ricker_br_inputs = np.zeros((batch_size, n_scales, time_length, 1))
    ricker_vr_inputs = np.zeros((batch_size, n_scales, time_length, 1))
    haar_br_inputs = np.zeros((batch_size, time_length, 1))
    
    for i in range(batch_size):
        # get individual samples
        if isinstance(ricker_br_batch, list):
            ricker_br_sample = ricker_br_batch[i]
            ricker_vr_sample = ricker_vr_batch[i]
            haar_br_sample = haar_br_batch[i]
        else:
            ricker_br_sample = ricker_br_batch[i]
            ricker_vr_sample = ricker_vr_batch[i]
            haar_br_sample = haar_br_batch[i]
        
        # process each sample
        # ricker br
        if ricker_br_sample.shape[1] != time_length:
            if ricker_br_sample.shape[1] > time_length:
                ricker_br_sample = ricker_br_sample[:, :time_length]
            else:
                pad_width = time_length - ricker_br_sample.shape[1]
                ricker_br_sample = np.pad(ricker_br_sample, ((0,0), (0, pad_width)), mode='edge')
        # ricker vr
        if ricker_vr_sample.shape[1] != time_length:
            if ricker_vr_sample.shape[1] > time_length:
                ricker_vr_sample = ricker_vr_sample[:, :time_length]
            else:
                pad_width = time_length - ricker_vr_sample.shape[1]
                ricker_vr_sample = np.pad(ricker_vr_sample, ((0,0), (0, pad_width)), mode='edge')

        # haar
        if len(haar_br_sample) != time_length:
            if len(haar_br_sample) > time_length:
                haar_br_sample = haar_br_sample[:time_length]
            else:
                pad_width = time_length - len(haar_br_sample)
                haar_br_sample = np.pad(haar_br_sample, (0, pad_width), mode='edge')
        
        # store in batch arrays
        ricker_br_inputs[i] = ricker_br_sample.reshape(n_scales, time_length, 1)
        ricker_vr_inputs[i] = ricker_vr_sample.reshape(n_scales, time_length, 1)
        haar_br_inputs[i] = haar_br_sample.reshape(time_length, 1)
    
    return ricker_br_inputs, ricker_vr_inputs, haar_br_inputs

def create_labels_from_annotations(
    time_length,
    switchback_periods,  # list of (start_time, end_time) tuples
    smoothing_window=3   # smooth labels around switchback boundaries
):
    """
    Creates training labels from switchback annotations.
    
    """
    labels = np.zeros(time_length)

    # the start and end
    for start, end in switchback_periods:
        start = max(0, start)
        end = min(time_length, end)
        labels[start:end] = 1
    
    # appply smoothing to handle boundary effects
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        labels = np.convolve(labels, kernel, mode='same')
        labels = (labels > 0.1).astype(float)  # Threshold back to binary
    
    return labels.reshape(time_length, 1)

def create_global_labels(local_labels_batch):
    """
    Creates global presence labels from local detection labels.
    """
    # flatten along time dimensions and check if any 1s exist
    if len(local_labels_batch.shape) == 4:  # (batch, scales, time, 1)
        global_labels = np.any(local_labels_batch > 0.5, axis=(1, 2)).astype(float)
    else:  # (batch, time, 1) or (batch, time)
        global_labels = np.any(local_labels_batch > 0.5, axis=1).astype(float)
    
    return global_labels.reshape(-1, 1)

def reshape_for_model(
    ricker_br_array,    # 2D: (n_scales, time_length)
    ricker_vr_array,    # 2D: (n_scales, time_length)  
    haar_br_array       # 1D: (time_length,)
):
    """
    Simple reshaping function for single input to model.predict()
    The whole thing will fuck up if not
    """
    # add batch and channel dimensions
    ricker_br_input = ricker_br_array.reshape(1, ricker_br_array.shape[0], ricker_br_array.shape[1], 1)
    ricker_vr_input = ricker_vr_array.reshape(1, ricker_vr_array.shape[0], ricker_vr_array.shape[1], 1)
    haar_br_input = haar_br_array.reshape(1, len(haar_br_array), 1)
    
    return [ricker_br_input, ricker_vr_input, haar_br_input]
