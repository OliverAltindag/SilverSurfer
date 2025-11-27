import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys

# --- IMPORT YOUR MODULES ---
import dataload
import wtransforms
import tensorcreator
import inputimages
import slidingwindow
import labelcreator
import CNN

DATA_FOLDER = 'Data'  
CSV_PATH = 'E08_PSP_switchback_event_list.csv' # change
TARGET_DATE = '2021-04-27'

# CHECK THIS: Look at your mag_rtn filename for this date.
# If it is 'psp_fld_l2_mag_rtn_2021042700_v02.cdf', the chunk is '00'.
MAG_CHUNK = '00' 
# ==========================================

def run_test_on_real_data():
    print(f"--- STEP 1: Loading Real Data for {TARGET_DATE} ---")
    
    # Load the raw CDF data using your dataload.py
    # This expects both MAG and SPC files to exist in DATA_FOLDER
    mag_data, spc_data = dataload.load_psp_data(DATA_FOLDER, TARGET_DATE, MAG_CHUNK)
    
    if mag_data is None or spc_data is None:
        print("CRITICAL ERROR: Could not load MAG or SPC data.")
        print(f"Ensure files for {TARGET_DATE} (Chunk {MAG_CHUNK}) are in '{DATA_FOLDER}'")
        sys.exit(1)

    t_mag, b_rtn = mag_data
    t_spc, v_rtn = spc_data
    
    print(f"Loaded Mag: {len(t_mag)} points")
    print(f"Loaded Spc: {len(t_spc)} points")

    print("\n--- STEP 2: Running Tensor Creator (Sync & Wavelets) ---")
    # Sync and Transform using your tensorcreator.py
    # This handles the interpolation and the Ricker/Haar math
    ricker_br, ricker_vr, haar_br, T_final = tensorcreator.create_feature_tensor(
        t_mag, b_rtn, t_spc, v_rtn
    )
    
    print(f"Tensor Shape Check:")
    print(f"Ricker Br: {ricker_br.shape} (Expected: 64, N)")
    print(f"Haar Br:   {haar_br.shape}   (Expected: N, )")
    print(f"T_final:   {len(T_final)} timestamps")

    print("\n--- STEP 3: Generating Labels from CSV ---")
    # Create Labels matching T_final using your labelcreator.py
    # This maps the start/end times from E08...csv to the T_final index
    labels = labelcreator.generate_labels_from_csv(T_final, CSV_PATH)
    
    # Validation: Did we find the event?
    num_positive = np.sum(labels)
    print(f"Total positive label points found: {int(num_positive)}")
    
    if num_positive == 0:
        print("WARNING: No switchbacks matched in this time range.")
        print("Check: 1. Is the date in the CSV? 2. Are CSV headers correct?")
    else:
        print(f"SUCCESS: Found events covering {num_positive * 0.013 / 60:.2f} minutes.")

    print("\n--- STEP 4: Creating Sliding Windows ---")
    # Slice into windows using your slidingwindow.py
    WINDOW_SIZE = 128
    STRIDE = 16 # Small stride to generate more training samples from this single day
    
    batch_r_br, batch_r_vr, batch_h_br, batch_lbls, Y_global = slidingwindow.create_sliding_windows(
        ricker_br, ricker_vr, haar_br, labels, 
        window_size=WINDOW_SIZE, stride=STRIDE
    )
    
    if batch_r_br is None:
        print("Error creating windows.")
        sys.exit(1)

    print(f"Created {batch_r_br.shape[0]} windows.")
    # Check shape consistency for CNN
    print(f"Input Shape: {batch_r_br.shape} (Batch, Scales, Time, 1)")

    print("\n--- STEP 5: Building Model ---")
    # \Build Model using your CNN.py
    ricker_shape = (batch_r_br.shape[1], batch_r_br.shape[2], 1)
    haar_shape = (batch_h_br.shape[1], 1)
    
    model = CNN.create_multi_input_switchback_cnn(
        ricker_shape=ricker_shape,
        haar_shape=haar_shape
    )

    print("\n--- STEP 6: Memorization Test (Overfitting this day) ---")
    # Train/Fit
    # We attempt to 'overfit' on this specific day to ensure the model 
    # can physically learn the patterns identified by your labels.
    
    # Callbacks to stop if it learns perfectly
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        x=[batch_r_br, batch_r_vr, batch_h_br],
        y={
            'final_flat_output': batch_lbls,
            'global_switchback_presence': Y_global
        },
        epochs=30, 
        batch_size=32,
        verbose=1,
        callbacks=[early_stop]
    )
    
    final_loss = history.history['loss'][-1]
    final_acc = history.history['global_switchback_presence_accuracy'][-1]
    
    print("\n=== TEST RESULTS ===")
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Global Accuracy: {final_acc:.4f}")
    
    if final_acc > 0.90:
        print("SUCCESS: Pipeline is verified. The model successfully memorized the real events.")
    else:
        print("INCONCLUSIVE: Accuracy is low. Ensure the CSV labels align with visible spikes in the data.")

if __name__ == "__main__":
    run_test_on_real_data()
