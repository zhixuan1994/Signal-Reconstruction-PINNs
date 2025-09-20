import numpy as np
import matplotlib.pyplot as plt
import SR_PINN
from tensorflow.keras import layers, models
from sklearn.impute import KNNImputer
from statsmodels.tsa.ar_model import AutoReg
import pandas as pd
from sklearn.metrics import mean_absolute_error

n = 200
t = np.linspace(0, 1, n)

# Case 1
signal = (t+0.1)*np.sin(t*np.pi*8)

# Case 2
# signal = np.sin(4*np.pi*t)/3
# signal[int(n/2):] = np.sin(4*np.pi*t[int(n/2):])

# LSTM data preprocess
def data_preprocess(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        if np.isnan(series[i:i+window_size]).any() or np.isnan(series[i+window_size]):
            continue 
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

# Add noise
sigma = 0.1
noise = np.random.normal(0, sigma, size=signal.shape)
noisy_signal = signal + noise

# Set missing values
mask = np.random.rand(len(signal)) > 0.3
signal_missing = noisy_signal.copy()
for i in range(10):
    if mask[i] == False:
        mask[i] = True
signal_missing[~mask] = np.nan

# TVAR-PINNs based
lags_r = 2
TV_PINN = SR_PINN.TVAR_PINN(lags_r, signal_missing, False)
TV_PINN.main()
sld_win = SR_PINN.sliding_windows(TV_PINN.model, lags_r, signal_missing, 20, 1)
sig_PINNs = sld_win.main()

# LSTM
window_size = 5
X, y = data_preprocess(signal_missing, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))
y = y.reshape((-1, 1))
model = models.Sequential([
    layers.LSTM(64, activation='tanh', return_sequences=True, input_shape=(window_size, 1)),
    layers.LSTM(32, activation='tanh'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=30, batch_size=32, verbose=1)

LSTM_sig = signal_missing.copy()
for i in range(window_size, len(signal)):
    if np.isnan(LSTM_sig[i]):
        input_seq = LSTM_sig[i-window_size:i]
        if np.isnan(input_seq).any():
            continue
        LSTM_sig[i] = model.predict(input_seq.reshape(1, window_size, 1), verbose=0)

# KNN
imputer = KNNImputer(n_neighbors=15, weights="distance")
KNN_sig = imputer.fit_transform(signal_missing.reshape(-1,1))
KNN_sig = KNN_sig.reshape(-1,)

# AR
signal_init = pd.Series(signal_missing).interpolate().to_numpy()
model = AutoReg(signal_init, lags=2, old_names=False).fit()
AR_sig = model.predict(start=0, end=len(signal_init)-1)
