import numpy as np
from moabb.datasets import Liu2024
from moabb.paradigms import MotorImagery
from sklearn.pipeline import Pipeline
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

import warnings

warnings.filterwarnings('ignore')

print("Loading dataset locally (this will be fast since it's already cached)...")
dataset = Liu2024()
paradigm = MotorImagery(n_classes=2, fmin=8.0, fmax=30.0, tmin=0.0, tmax=3.0)

# Loading Subjects 1, 2, 3 as Base, and Subject 4 as Target
X, y, meta = paradigm.get_data(dataset=dataset, subjects=[1, 2, 3, 4])
groups = meta['subject'].values


target_subject = 4
base_idx = (groups != target_subject)
target_idx = (groups == target_subject)

X_base, y_base = X[base_idx], y[base_idx]
X_target_all, y_target_all = X[target_idx], y[target_idx]

# 10-Shot Calibration
X_calib, _, y_calib, _ = train_test_split(
    X_target_all, y_target_all, train_size=10, stratify=y_target_all, random_state=42
)

X_combined = np.vstack((X_base, X_calib))
y_combined = np.hstack((y_base, y_calib))

# Sample Weights
weights_base = np.ones(len(y_base))
weights_calib = np.ones(len(y_calib)) * 10.0
sample_weights = np.hstack((weights_base, weights_calib))

# Build and Train Pipeline
full_pipeline = Pipeline([
    ('cov', Covariances(estimator='oas')),
    ('ts', TangentSpace(metric='logeuclid')),
    ( 'scaler',StandardScaler()),
    ('clf', RidgeClassifier())
])

print("Training Few-Shot pipeline...")
full_pipeline.fit(X_combined, y_combined, clf__sample_weight=sample_weights)

# Save native model
joblib.dump(full_pipeline, 'stroke_bci_full_pipeline.pkl')
print("Native Windows model saved as 'stroke_bci_full_pipeline.pkl'!")