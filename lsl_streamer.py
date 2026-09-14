import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet
from moabb.datasets import Liu2024
import mne

mne.set_log_level('WARNING')
print("Downloading/Loading Subject 4 dataset...")
dataset = Liu2024()

subject_data = dataset.get_data(subjects=[4])[4]
first_session_key = list(subject_data.keys())[0]
session_data = subject_data[first_session_key]
first_run_key = list(session_data.keys())[0]
raw_data = session_data[first_run_key]


raw_data.load_data()


raw_data.pick_types(eeg=True)


print("Applying 8-30 Hz bandpass filter to the continuous stream...")
raw_data.filter(l_freq=8.0, h_freq=30.0)


data_array = raw_data.get_data().T

srate = 500  
n_channels = data_array.shape[1]

    
print(f"Successfully loaded data from {first_session_key} -> {first_run_key}")


print(data_array)
print("Configuring LSL Stream...")


info = StreamInfo('Liu_Stroke_Sim', 'EEG', n_channels, srate, 'float32', 'sim_subj_4')


outlet = StreamOutlet(info)
print("LSL Outlet opened. Broadcasting data...")

sample_duration = 1.0 / srate  

try:
    for i in range(data_array.shape[0]):
        start_time = time.time()
        
       
        outlet.push_sample(data_array[i, :])
        
       
        elapsed_time = time.time() - start_time
        sleep_time = sample_duration - elapsed_time
        
        if sleep_time > 0:
            time.sleep(sleep_time)
            
except KeyboardInterrupt:
    print("\nStreaming stopped by user.")