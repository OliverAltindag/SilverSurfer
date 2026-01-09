import numpy as np
import pandas as pd
from datetime import timedelta
import wtransforms as wt

def create_feature_tensor(t_mag, b_rtn, t_spc, v_rtn, window_size=128, return_raw=False):
    """
    Syncs B and V data, cleans V_R, calculates Ricker and Haar features,
    and returns a single, continuous 3-channel tensor.
    If return_raw is True, also returns the normalized 1D time series (B_final, V_final).
    """
    

    # Replace Fill Values (-1e31) AND Infinities with NaN
    # the encounter one data was way shittier than I anticipated
    # dog eat dog
    # improvise adapt overcome
    b_rtn = np.where(b_rtn < -1e9, np.nan, b_rtn)
    b_rtn = np.where(~np.isfinite(b_rtn), np.nan, b_rtn)

    v_rtn = np.where(v_rtn < -1e9, np.nan, v_rtn)
    v_rtn = np.where(~np.isfinite(v_rtn), np.nan, v_rtn)


    # isolate V_R (Radial component)
    v_r_raw = v_rtn[:, 0]
    
    # create df for V_R indexed by SPC time
    df_vel = pd.DataFrame({'V_R': v_r_raw}, index=t_spc)
    df_vel = df_vel[~df_vel.index.duplicated(keep='first')]

    # reindex V_R data onto the faster B_R timeline
    # all of this crumbles if not :(
    t_mag_dt = pd.Series(t_mag)
    df_vel_synced = df_vel.reindex(t_mag_dt, method='nearest', tolerance=timedelta(seconds=1))
    
    # extract the synced V_R signal
    v_signal_synced = df_vel_synced['V_R'].values

    # apply Robust Fit to Synced V_R
    # (The loose fit will now smoothly bridge the NaNs we created above)
    v_clean = wt.apply_loose_fit(v_signal_synced)


    # Solar wind velocity is physically between 200 and 1000 km/s
    # so we clamp it to -50 to 50 sigma
    v_median = np.nanmedian(v_clean)
    q75, q25 = np.percentile(v_clean[~np.isnan(v_clean)], [75 ,25])
    iqr = q75 - q25
    if iqr < 1e-5: iqr = 1.0
    
    v_clean = np.nan_to_num(v_clean, nan=v_median) # Replace remaining NaNs
    v_clean = (v_clean - v_median) / iqr
    
    # HARD CLAMP: Ensure no value exceeds +/- 50 sigma 
    # really didnt want the training to blow up as I slept
    # it didnt but got through a shocking 0.5 of one of 10+ encounters
    # so currently more of a hypothetical good model
    # sunject to change
    v_clean = np.clip(v_clean, -50.0, 50.0)
   
    # B_R: Get B_R and normalize it to unit vector
    b_r_raw = b_rtn[:, 0]
    b_mag = np.linalg.norm(b_rtn, axis=1)
    
    # Handle NaNs in B_Mag
    b_mag = np.nan_to_num(b_mag, nan=1e-5)
    
    B_normalized = b_r_raw / np.maximum(b_mag, 1e-5)
    
    # loosely fit it t B as well
    # This prevents the "Zero Square Wave" artifacts by interpolating gaps.
    B_normalized = wt.apply_loose_fit(B_normalized)
    # -------------------------------------------


    # B_normalized should be between -1 and 1.
    # clip it to -1.1 to 1.1 to be safe and catch artifacts.
    B_normalized = np.nan_to_num(B_normalized, nan=0.0) 
    B_normalized = np.clip(B_normalized, -1.1, 1.1)


    # create DataFrames to align exact timestamps
    df_B = pd.DataFrame({'B': B_normalized}, index=t_mag)
    df_V = pd.DataFrame({'V': v_clean}, index=t_mag_dt)
    
    # Inner Join on Index (Timestamps)
    df_merged = df_B.join(df_V, how='inner', lsuffix='_B', rsuffix='_V')
    
    # extract perfectly aligned arrays
    B_final = df_merged['B'].values
    V_final = df_merged['V'].values
    T_final = df_merged.index.values

    # =========================================================
    # RIGOROUS TIME NORMALIZATION (The "Universal Clock Fix")
    # Enforces 14.5 Hz sampling rate across ALL files
    # kinda arbitrary sampling rate but it works
    # =========================================================
    
    TARGET_DT = 0.068965517  # 1.0 / 14.5 Hz (in seconds)
    
    if len(T_final) > 1:
        # Convert numpy datetime64 to float seconds since epoch
        t_numeric = T_final.astype('datetime64[ns]').astype(np.float64) / 1e9
        
        # Create the perfect numeric grid
        t_new_numeric = np.arange(t_numeric[0], t_numeric[-1], TARGET_DT)
        
        # Interpolate B and V onto the perfect grid
        B_final = np.interp(t_new_numeric, t_numeric, B_final)
        V_final = np.interp(t_new_numeric, t_numeric, V_final)
        
        # Convert back to datetime64
        T_final = (t_new_numeric * 1e9).astype('datetime64[ns]')


    # Wavelet Transforms
    ricker_br = wt.get_ricker_features_fast(B_final, scales=np.arange(1, 65))
    ricker_vr = wt.get_ricker_features_fast(V_final, scales=np.arange(1, 65))
    haar_br = wt.get_haar_features(B_final)

    # Ensure absolutely no NaNs leave this function
    # these were my enemy in training
    ricker_br = np.nan_to_num(ricker_br, nan=0.0)
    ricker_vr = np.nan_to_num(ricker_vr, nan=0.0)
    haar_br = np.nan_to_num(haar_br, nan=0.0)

    if return_raw:
        return ricker_br, ricker_vr, haar_br, T_final, B_final, V_final
    else:
        return ricker_br, ricker_vr, haar_br, T_final
