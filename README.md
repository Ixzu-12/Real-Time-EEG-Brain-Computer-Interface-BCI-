# Real-Time-EEG-Brain-Computer-Interface-BCI-
Backend neuro-engineering system for live continuous EEG processing. Uses Few-Shot Transfer Learning and DSP to decode physical motor imagery for real-world hardware actuation.
Repository Topics (Tags):
bci

eeg

digital-signal-processing

machine-learning

neuro-engineering

python

lsl
# Real-Time EEG Brain-Computer Interface (BCI)

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Active-orange?logo=scikit-learn)
![MNE](https://img.shields.io/badge/MNE-Neuroscience-red)
![LSL](https://img.shields.io/badge/Networking-Lab_Streaming_Layer-lightgrey)

A live digital signal processing (DSP) pipeline and Few-Shot Transfer Learning architecture for continuous motor imagery classification. This backend system broadcasts 29-channel EEG data over a Lab Streaming Layer (LSL) network, resolving continuous Riemannian geometry calculations in real-time to decode Left/Right physical intention.

## 🧠 System Architecture

This project bypasses static dataset evaluation to build a true continuous-time inference engine. 

```mermaid
graph TD
    A[Mock EEG Hardware / Liu2024 Dataset] -->|Continuous Signal| B(MNE DSP: 8-30 Hz Bandpass)
    B -->|29-Channel Array| C[LSL Streamer]
    C -->|Local WiFi Network| D[LSL Receiver]
    D -->|2-Second Sliding Window| E(PyRiemann: Covariances)
    E -->|Log-Euclidean Metric| F(PyRiemann: Tangent Space)
    F -->|Raw Vector Features| G{Ridge Classifier}
    G -->|Threshold > 0.8| H[RIGHT HAND INTENT]
    G -->|Threshold < -0.8| I[LEFT HAND INTENT]
    G -->|Threshold between -0.8 and 0.8| J[RESTING / NOISE]
⚙️ Core Engineering Challenges Solved
The "Null Class" (Resting State) Problem: Standard BCI models force binary predictions even when a patient is resting. This architecture uses the Ridge Classifier's native decision_function() to establish a rigid mathematical confidence threshold (+/- 0.8). If the signal-to-noise ratio drops, the system correctly defaults to a safe "Resting" state.

DSP Edge Artifacts: Applying IIR filters to 2-second live chunks causes mathematical ringing that destroys covariance matrices. This pipeline solves this by applying a continuous 8-30 Hz Butterworth bandpass filter upstream at the streamer level, feeding mathematically stable data to the network.

Cross-Subject Distribution Shifts: Brainwaves differ radically between patients, especially stroke victims. Instead of deep learning, this system utilizes a 10-Shot Calibration Pipeline leveraging PyRiemann Tangent Space mapping to adapt to unseen patient variance natively with minimal training data.

📂 Repository Structure
train_local.py: Offline pipeline builder. Downloads the Liu2024 dataset, isolates target and base subjects, applies sample weighting, and exports the native Scikit-Learn pipeline (stroke_bci_full_pipeline.pkl).

lsl_streamer.py: The mock hardware broadcast node. Ingests raw data, isolates 29 EEG channels, applies the continuous 8-30 Hz filter, and pushes 500 Hz packets to the local network.

lsl_receiver.py: The live inference engine. Hooks into the LSL network, manages a 2.0-second sliding buffer, calculates real-time Riemannian distances without scaling multipliers, and prints terminal predictions.

🚀 Quickstart Guide
1. Install Dependencies

Bash
pip install mne pyriemann scikit-learn pylsl moabb
2. Calibrate the Pipeline
Train the native Windows model to generate the .pkl file.

Bash
python train_local.py
3. Initialize the Real-Time Network
Open two separate terminals to simulate the hardware and the decoder.

Terminal 1 (Hardware Broadcast):

Bash
python lsl_streamer.py
Terminal 2 (Live Inference):

Bash
python lsl_receiver.py
🔌 Next Steps: Hardware Actuation
Because this system outputs highly stable terminal predictions based on continuous physical intent, the lsl_receiver.py output can be routed via PySerial to a microcontroller (e.g., Arduino/ESP32) to physically trigger relays, servo motors, or environmental controls.
