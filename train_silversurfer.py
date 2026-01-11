"""
SilverSurfer Training Script
============================
User-Friendly training script for the SilverSurfer switchback detection model.

INSTRUCTIONS:
1. Set your Paths in the CONFIGURATION section below.
2. Run this script: python train_silversurfer.py
"""

import numpy as np
import tensorflow as tf
import os
import glob
import gc
import cdflib
import pandas as pd
import random
import argparse

# --- IMPORT MODULES ---
import dataload
import tensorcreator
import slidingwindow
import labelcreator
import CNN

# =========================================
# USER CONFIGURATION (EDIT THESE)
# =========================================

# Path to the folder containing your .cdf files (MAG and SPC)
DATA_FOLDER = 'Data'                     

# Path to your manually labeled event list (CSV)
CSV_PATH = 'E06_PSP_switchback_event_list.csv'

# Where to save the model weights
WEIGHTS_FILE = 'SilverSurfer_weights.weights.h5'

# Training Parameters
EPOCHS_PER_FILE = 5
LEARNING_RATE = 1e-5

# =========================================
# SYSTEM PARAMETERS (DO NOT EDIT)
# Pretty Please

# Cherry
# Top
# =========================================
# These parameters are tuned for the physics of Encounter 6.
# Changing them requires retraining the entire architecture.
WINDOW_SIZE = 256  # 256 time steps (approx 18 seconds at 14.5Hz)
STRIDE = 64        # 75% Overlap
BATCH_SIZE = 32    # Optimal for VRAM usage

def main():
    print("\n" + "="*40)
    print("SILVERSURFER TRAINING")
    print("="*40 + "\n")

    # Inputs
    # Data Folder
    default_data = 'Data'
    user_data = input(f"Enter Data Folder path [default: '{default_data}']: ").strip()
    data_folder = user_data if user_data else default_data

    # CSV Path
    default_csv = 'switchback_event_list.csv'
    user_csv = input(f"Enter Label CSV path [default: '{default_csv}']: ").strip()
    csv_path = user_csv if user_csv else default_csv

    # Weights File
    default_weights = 'SilverSurfer_weights.weights.h5'
    # I mean this is what the code automatically saves so like
    # Not gonna make interactive bc they would just be doing more work
    weights_path = default_weights 

    # Epochs
    default_epochs = 7
    user_epochs = input(f"Enter Epochs per file [default: {default_epochs}]: ").strip()
    epochs = int(user_epochs) if user_epochs else default_epochs

    # Learning Rate
    default_lr = 1e-4
    user_lr = input(f"Enter Learning Rate [default: {default_lr}]: ").strip()
    lr = float(user_lr) if user_lr else default_lr

    print("\n" + "-"*30)
    print("SUMMARY:")
    print(f"Data:    {data_folder}")
    print(f"Labels:  {csv_path}")
    print(f"Epochs:  {epochs}")
    print(f"LR:      {lr}")
    print("-"*30 + "\n")
    
    confirm = input("Start Training? (y/n): ").lower()
    if confirm != 'y':
        print("Aborted.")
        return

    # setup the model
    RICKER_SHAPE = (64, WINDOW_SIZE, 1) 
    HAAR_SHAPE = (WINDOW_SIZE, 1)
    
    print("Building Model...")
    model = CNN.create_multi_input_switchback_cnn(
        ricker_shape=RICKER_SHAPE,
        haar_shape=HAAR_SHAPE
    )
    
    # Re-compile with user Learning Rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    if hasattr(model.optimizer, 'learning_rate'):
        model.optimizer.learning_rate.assign(lr)
    else:
        # If it wasn't compiled, compile it. 
        # But create_multi_input_switchback_cnn likely compiles.
        pass

    if os.path.exists(weights_path):
        print(f"Loading existing weights from {weights_path}...")
        try:
            model.load_weights(weights_path)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load weights ({e}). Starting fresh.")
    else:
        print("No existing weights found. Starting fresh.")

    # find the files
    mag_files = glob.glob(os.path.join(data_folder, '*mag*.cdf'))
    random.shuffle(mag_files)
    
    spc_files = sorted(glob.glob(os.path.join(data_folder, '*spc*.cdf')))
    
    if not mag_files:
        print("No MAG files found!")
        return

    # main loopage
    for mag_path in mag_files:
        filename = os.path.basename(mag_path)
        print(f"\nProcessing: {filename}")
        
        try:
            # load data
            cdf_mag = cdflib.CDF(mag_path)
            # Try/Except for variable names
            try:
                t_mag = dataload.convert_cdf_time(cdf_mag.varget("epoch_mag_RTN"))
                b_rtn = cdf_mag.varget("psp_fld_l2_mag_RTN")
            except:
                t_mag = dataload.convert_cdf_time(cdf_mag.varget("epoch_mag_RTN_1min"))
                b_rtn = cdf_mag.varget("psp_fld_l2_mag_RTN_1min")
            
            # Find SPC
            parts = filename.split('_')
            date_part = next((p for p in parts if p.startswith('20') and len(p) >= 8), None)
            date_key = date_part[:8] if date_part else ""
            matching_spc = [f for f in spc_files if date_key in f]
            
            if not matching_spc:
                print(f"SKIP: No SPC file for {date_key}")
                continue
            
            cdf_spc = cdflib.CDF(matching_spc[0])
            info = cdf_spc.cdf_info()
            z_vars = info['zVariables'] if isinstance(info, dict) else info.zVariables
            v_var = "vp_moment_RTN" if "vp_moment_RTN" in z_vars else "vp_fit_RTN"
            t_var = "Epoch" if "Epoch" in z_vars else "epoch"
            
            t_spc = dataload.convert_cdf_time(cdf_spc.varget(t_var))
            v_rtn = cdf_spc.varget(v_var)
            
            # features
            ricker_br, ricker_vr, haar_br, T_final = tensorcreator.create_feature_tensor(
                t_mag, b_rtn, t_spc, v_rtn
            )
            
            # labels
            labels = labelcreator.generate_labels_from_csv(T_final, csv_path)
            if np.sum(labels) == 0:
                print("No events in this file. Skipping.")
                continue

            # windows
            X_r_br, X_r_vr, X_h_br, Y_local, Y_global = slidingwindow.create_sliding_windows(
                ricker_br, ricker_vr, haar_br, labels,
                window_size=WINDOW_SIZE, stride=STRIDE
            )
            
            if X_r_br is None or len(X_r_br) == 0:
                continue

            # train
            indices = np.arange(len(X_r_br))
            np.random.shuffle(indices)
            
            model.fit(
                [X_r_br[indices], X_r_vr[indices], X_h_br[indices]],
                [Y_local[indices], Y_global[indices]],
                batch_size=BATCH_SIZE,
                epochs=epochs,
                verbose=1
            )
            
            # save
            model.save_weights(weights_path)
            print(f"Saved weights to {weights_path}")
            
            # Cleanup
            del t_mag, b_rtn, t_spc, v_rtn, ricker_br, labels, X_r_br
            gc.collect()
            
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()
