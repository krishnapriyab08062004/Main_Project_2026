import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load training features used earlier
X_train = np.load("data/processed/X_train.npy")   # shape: (samples, 128, 101)

# Reshape for scaler
samples, time, features = X_train.shape
X_reshaped = X_train.reshape(-1, features)

# Fit scaler
scaler = StandardScaler()
scaler.fit(X_reshaped)

# Save scaler
with open("data/processed/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("scaler.pkl created successfully")
