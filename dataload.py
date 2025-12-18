import cdflib
import numpy as np
import pandas as pd
import glob
import os
from datetime import datetime

def convert_cdf_time(raw_times):
    """
    Converts CDF TT2000 nanosecond timestamps to Pandas Datetime objects.
    This is critical to the time-series data and the data transoformations that will follow.
    These will need to be standardized to the training data as there is an offset in each 
    new data release. 

    I kind of assumed it was in all of them and it seemed to work unless it just got super lucky,
    but unlikely.
    """
    # TT2000 base is 2000-01-01 12:00:00
    base_time = pd.Timestamp("2000-01-01 12:00:00")
    raw_arr = np.array(raw_times, dtype='int64')
    
    # apply the Leap Second Correction, if not the trianing labels will be off 
    # TT2000 is currently ~69.184 seconds AHEAD of UTC 
    # does not need to be perfection but close enough
    tt2000_dt = base_time + pd.to_timedelta(raw_arr, unit='ns')
    return tt2000_dt - pd.Timedelta(seconds=69.184)

def load_psp_data(data_folder, target_date, mag_chunk):
    """
    Loads Magnetic Field (FIELDS) and Plasma (SWEAP) data for a specific date.
    It needs the Br, Bt, Bn data.
    It also needs the radial winds data to prepare the data for training.

    mag_chunk is the 0.6hr time blocks in the filenames from parker: REMEMBER THIS.
    """
    # cleans what is rather ugly data
    date_str_clean = target_date.replace('-', '')
    
    # construct the general file fatterns for them both
    mag_pattern = os.path.join(data_folder, f"*mag_rtn*{date_str_clean}{mag_chunk}*.cdf")
    spc_pattern = os.path.join(data_folder, f"*spc*{date_str_clean}*.cdf")
    
    # gets the neccesary files
    mag_files = glob.glob(mag_pattern)
    spc_files = glob.glob(spc_pattern)
    
    # loads the magnetic field data
    # has a proper fallback shoud the data not exist
    mag_data = None
    if mag_files:
        print(f"Found MAG file: {os.path.basename(mag_files[0])}")
        try:
            cdf_mag = cdflib.CDF(mag_files[0])
            t_mag = convert_cdf_time(cdf_mag.varget("epoch_mag_RTN"))
            b_rtn = cdf_mag.varget("psp_fld_l2_mag_RTN")
            mag_data = (t_mag, b_rtn)
        except Exception as e:
            print(f"Error reading MAG file: {e}")
    else:
        print(f"No MAG file found for {target_date} (Chunk {mag_chunk})")

    # loads in the plasma velocity
    # includes fallback for missing/corruped files
    spc_data = None
    if spc_files:
        print(f"Found SPC file: {os.path.basename(spc_files[0])}")
        try:
            cdf_spc = cdflib.CDF(spc_files[0])
            
            # attempt to find the correct variable names (they vary by version) :(
            # we prioritize moment data over fit data
            # the data needs to be untouched and raw
            # even if this is noiser, and harder to work with
            v_var = None
            t_var = None
            
            info = cdf_spc.cdf_info()
            if isinstance(info, dict): z_vars = info['zVariables']
            else: z_vars = info.zVariables # For newer cdflib versions

            # check for Velocity
            if "vp_moment_RTN" in z_vars: v_var = "vp_moment_RTN"
            elif "vp_fit_RTN" in z_vars: v_var = "vp_fit_RTN"
            
            # check for Time
            if "Epoch" in z_vars: t_var = "Epoch"
            elif "epoch" in z_vars: t_var = "epoch"

            # convert to the proper time using the helper function defined above
            if v_var and t_var:
                t_spc = convert_cdf_time(cdf_spc.varget(t_var))
                v_rtn = cdf_spc.varget(v_var)
                spc_data = (t_spc, v_rtn)
            else:
                print("Could not find required SPC variables (Epoch/Velocity).")
                
        except Exception as e:
            print(f"Error reading SPC file: {e}")
    else:
        print(f"No SPC file found for {target_date}")

    return mag_data, spc_data
