import time
import numpy as np
from pylsl import StreamInlet, resolve_byprop
import joblib
import warnings
import sys


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



print("\n--- Starting Real-Time Decoder ---")

try:
    while True:
        chunk, timestamps = inlet.pull_chunk(max_samples=srate)
        
        if chunk:
            data_buffer = np.vstack([data_buffer, chunk])
            
            if len(data_buffer) >= buffer_size:
              
                window = data_buffer[-buffer_size:, :]
                
                
                
               
                X_window = window.T.reshape(1, n_channels, buffer_size)
                X_window_scaled = X_window * 1e6
               
                
                try:
                    decision_score = clf.decision_function(X_window_scaled)[0]
                    THRESHOLD = 0.8
                    
                    if decision_score < -THRESHOLD: 
                        print(f">> Predicted Intention: [LEFT HAND]  (Score: {decision_score:.2f}) <<")
                    elif decision_score > THRESHOLD:
                        print(f">> Predicted Intention: [RIGHT HAND] (Score: {decision_score:.2f}) <<")
                    else:
                        print(f"-- Resting / Processing... (Score: {decision_score:.2f}) --")
                        
                except ValueError as e:
                    print(f"Shape Mismatch Error: {e}")
                    import sys
                    sys.exit()

                
                overlap = int(srate * 1.5)
                data_buffer = data_buffer[-overlap:, :]
                
        
        time.sleep(0.01)
        
except KeyboardInterrupt:
    print("\nReceiver stopped by user.")