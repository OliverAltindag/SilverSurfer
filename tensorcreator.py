import numpy as np
import pandas as pd
from datetime import timedelta
import wtransforms as wt

def create_feature_tensor(t_mag, b_rtn, t_spc, v_rtn, window_size=128):
    """
    Syncs B and V data, cleans V_R, calculates Ricker and Haar features,
    and returns a single, continuous 3-channel tensor (for one full time segment).
    
    This function should be run before the sliding window loop.
    """
    # interpolate Plasma (V_R, radial winds) onto Magnetic Field Time (T_MAG)
    # isolate V_R (Radial component)
    # does not work if they are not on the same time scale
    v_r_raw = v_rtn[:, 0]
    
    # create df for V_R indexed by SPC time
    df_vel = pd.DataFrame({'V_R': v_r_raw}, index=t_spc)
    df_vel = df_vel[~df_vel.index.duplicated(keep='first')] # Drop duplicates if present

    # reindex V_R data onto the faster B_R timeline (Nearest neighbor sync)
    # use a tolerance for safety, but data syncs mostly by time index matching
    t_mag_dt = pd.Series(t_mag) # Convert to series for easier reindexing
    df_vel_synced = df_vel.reindex(t_mag_dt, method='nearest', tolerance=timedelta(seconds=1))
    
    # extract the synced V_R signal
    v_signal_synced = df_vel_synced['V_R'].values

    # apply Robust Fit to Synced V_R
    v_clean = wt.apply_loose_fit(v_signal_synced)

    # normalizes the data so the model cna comprehend the gradients better
    # there are no disadvantages to this scientifically
    # should be strong enough to resist influence by switchbacks
    v_median = np.nanmedian(v_clean)
    q75, q25 = np.percentile(v_clean, [75 ,25])
    iqr = q75 - q25
    if iqr < 1e-5: iqr = 1.0
    v_clean = (v_clean - v_median) / iqr
    
    # B_R: Get B_R and normalize it to unit vector for CNN
    # this is CRITICAl for the information to be physically correct
    # it is also the norm for this process
    # and is how the whole thresholding became a thing
    # cite: Huang et al.
    b_r_raw = b_rtn[:, 0]
    b_mag = np.linalg.norm(b_rtn, axis=1)
    B_normalized = b_r_raw / np.maximum(b_mag, 1e-5) # Prevent divide by zero

    # length matching
    # The convolution and MRA math requires the input signal to have the same length
    # truncate the start of the shortest clean signal and match all arrays
    # shae but neccesary evil
    # create DataFrames to align exact timestamps
    df_B = pd.DataFrame({'B': B_normalized}, index=t_mag)
    df_V = pd.DataFrame({'V': v_clean}, index=t_mag_dt) # t_mag_dt calculated earlier
    
    # inner Join on Index (Timestamps)
    # Tthis drops points where EITHER sensor is missing, ensuring strict alignment
    df_merged = df_B.join(df_V, how='inner', lsuffix='_B', rsuffix='_V')
    
    # extract perfectly aligned arrays
    B_final = df_merged['B'].values
    V_final = df_merged['V'].values
    T_final = df_merged.index.values # this is now the Master Time for Label Generation

    # Downsample by 10 bc was ridiculously fine before
    # Increases window context to ~4.3s and speeds up training 10x
    B_final = B_final[::10]
    V_final = V_final[::10]
    T_final = T_final[::10]

    # puts these into wavelet form using the fucntions previously outlined
    ricker_br = wt.get_ricker_features_fast(B_final, scales=np.arange(1, 65))
    ricker_vr = wt.get_ricker_features_fast(V_final, scales=np.arange(1, 65))
    haar_br = wt.get_haar_features(B_final)

    return ricker_br, ricker_vr, haar_br, T_final
