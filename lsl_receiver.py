import time
import numpy as np
from pylsl import StreamInlet, resolve_byprop
import joblib
import warnings
import sys
import tkinter as tk

warnings.filterwarnings('ignore')

print("Looking for the 'Liu_Stroke_Sim' stream...")
streams = resolve_byprop('name', 'Liu_Stroke_Sim', timeout=10) 

if not streams:
    print("Error: Could not find the stream. Make sure lsl_streamer.py is actively running.")
    sys.exit()
    
inlet = StreamInlet(streams[0])
stream_info = inlet.info()
srate = int(stream_info.nominal_srate())
n_channels = stream_info.channel_count()

print(f"Connected to stream. Receiving {n_channels} channels at {srate} Hz.")
print("Loading Few-Shot Transfer Model...")
clf = joblib.load('stroke_bci_full_pipeline.pkl')
print("Model loaded successfully.")

window_size_seconds = 2.0
buffer_size = int(window_size_seconds * srate)
data_buffer = np.empty((0, n_channels))

# --- Native Tkinter UI Setup ---
root = tk.Tk()
root.title("Stroke BCI: Clinical Dashboard")
root.geometry("650x250")
root.configure(bg="#121212")

status_lbl = tk.Label(root, text="Calibrating...", font=("Consolas", 36, "bold"), fg="white", bg="#121212")
status_lbl.pack(expand=True, fill="both")

score_lbl = tk.Label(root, text="Decision Score: 0.00", font=("Consolas", 16), fg="#888888", bg="#121212")
score_lbl.pack(pady=10)

print("\n--- Starting Real-Time Decoder & UI ---")

try:
    while True:
        chunk, timestamps = inlet.pull_chunk()
        
        if chunk:
            data_buffer = np.vstack([data_buffer, chunk])
            
            if len(data_buffer) >= buffer_size:
                
                window = data_buffer[-buffer_size:, :]
                
                # EXACT WORKING MATH: No scaling, raw window reshape
                X_window = window.T.reshape(1, n_channels, buffer_size)
                
                try:
                    decision_score = float(clf.decision_function(X_window)[0])
                    THRESHOLD = 0.8
                    
                    if decision_score < -THRESHOLD: 
                        print(f">> Predicted Intention: [LEFT HAND]  (Score: {decision_score:.2f}) <<")
                        status_lbl.config(text="INTENTION: LEFT HAND", fg="#ff4444")
                    elif decision_score > THRESHOLD:
                        print(f">> Predicted Intention: [RIGHT HAND] (Score: {decision_score:.2f}) <<")
                        status_lbl.config(text="INTENTION: RIGHT HAND", fg="#4488ff")
                    else:
                        print(f"-- Resting / Processing... (Score: {decision_score:.2f}) --")
                        status_lbl.config(text="RESTING / NOISE", fg="#aaaaaa")
                        
                    # Update secondary score text on the UI
                    score_lbl.config(text=f"Decision Score: {decision_score:.2f}")
                        
                except ValueError as e:
                    print(f"Shape Mismatch Error: {e}")
                    sys.exit()

                # Slide the window forward by 0.5 seconds
                overlap = int(srate * 1.5)
                data_buffer = data_buffer[-overlap:, :]
                
        # Instantly update the UI graphics without pausing the LSL loop
        try:
            root.update()
        except tk.TclError:
            print("\nDashboard closed by user. Exiting...")
            sys.exit()
            
        # Yield the CPU briefly
        time.sleep(0.01)
        
except KeyboardInterrupt:
    print("\nReceiver stopped by user.")