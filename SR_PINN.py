import tensorflow as tf
import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

class LSTMDensePINN(tf.keras.Model):
    def __init__(self, lags_r):
        super(LSTMDensePINN, self).__init__()
        self.lstm1 = tf.keras.layers.LSTM(64, activation='tanh')
        self.lstm2 = tf.keras.layers.LSTM(32, activation='tanh')
        self.out = tf.keras.layers.Dense(lags_r + 1, activation='linear')
        
    def call(self, X):
        # X shape: (batch, 2)
        X_seq = tf.expand_dims(X, axis=1)  # add timestep dimension
        h = self.lstm1(X_seq)
        h = tf.expand_dims(h, axis=1)
        h = self.lstm2(h)
        u = self.out(h)
        return u
    
class TVAR_PINN():
    def __init__(self, lags_r, signal_missing, process_true):
        self.lags_r = lags_r
        self.obs_signal = signal_missing
        self.process_true = process_true

    def TV_AR_lagged_one(self, y_t):
        y_out = []
        for i in range(len(y_t) - self.lags_r):
            y_temp = np.concatenate([[1], y_t[i: self.lags_r+i, :].reshape(-1,)])
            y_out.append(y_temp)
        return np.array(y_out), y_t[self.lags_r:]

    def TV_AR_lagged_main(self, series):
        if np.sum(np.isnan(series)) == 0:
            y_out, y_target = self.TV_AR_lagged_one(series.reshape(-1,1))
            return y_out, y_target, np.arange(self.lags_r, len(series),1).reshape(-1,1) + self.lags_r
        series_segments, time_index = [], []
        start = 0
        series_ind = np.where(~np.isnan(series))[0]
        for i in range(1, len(series_ind)):
            if series_ind[i] != series_ind[i-1] + 1:  # break in continuity
                if len(series_ind[start:i]) > self.lags_r:
                    series_segments.append(series_ind[start:i])
                    time_index.append(series_ind[start:i-self.lags_r])
                start = i
        if len(series_ind[start:]) > self.lags_r:
            series_segments.append(series_ind[start:])
            time_index.append(series_ind[start:-self.lags_r])
        y_out, y_target = self.TV_AR_lagged_one(series[series_segments[0]].reshape(-1,1))
        time_out = time_index[0]
        for i in range(1, len(series_segments)):
            segment = series_segments[i]
            y_out_temp, y_target_temp = self.TV_AR_lagged_one(series[segment].reshape(-1,1))
            y_out = np.concatenate([y_out, y_out_temp], axis=0)
            y_target = np.concatenate([y_target, y_target_temp], axis=0)
            time_out = np.concatenate([time_out, time_index[i]])
        return y_out, y_target, time_out.reshape(-1,1) + self.lags_r

    def TV_AR_loss(self, right, left, TV_AR_coef):
        right = tf.reduce_sum(TV_AR_coef * right, axis=1)
        return tf.reduce_mean(tf.square(left.reshape(-1,) - right))
    
    def main(self):
        model = LSTMDensePINN(lags_r=self.lags_r)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.0, nesterov=False)
        signal_init = pd.Series(self.obs_signal).interpolate().to_numpy()

        epochs = 1000
        right, left, t_list= self.TV_AR_lagged_main(signal_init)
        
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                TV_AR_coef = model(t_list)
                TV_AR_coef = tf.reshape(TV_AR_coef, (TV_AR_coef.shape[0], -1))
                loss = self.TV_AR_loss(right, left, TV_AR_coef)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            if epoch % 50 == 0:
                if self.process_true == True:
                    print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")
        self.model = model


class sliding_windows():
    def __init__(self, model, lags_r, signal_missing, win_leng, sigma_w):
        self.model = model
        self.lags_r = lags_r
        self.obs_signal = signal_missing
        self.win_leng = win_leng
        self.sigma_w = sigma_w

    def TV_AR_lagged_one(self, y_t):
        y_out = []
        for i in range(len(y_t) - self.lags_r):
            y_temp = np.concatenate([[1], y_t[i: self.lags_r+i, :].reshape(-1,)])
            y_out.append(y_temp)
        return np.array(y_out), y_t[self.lags_r:]

    def TV_AR_lagged_main(self, series):
        if np.sum(np.isnan(series)) == 0:
            y_out, y_target = self.TV_AR_lagged_one(series.reshape(-1,1))
            return y_out, y_target, np.arange(self.lags_r, len(series),1).reshape(-1,1) + self.lags_r
        series_segments, time_index = [], []
        start = 0
        series_ind = np.where(~np.isnan(series))[0]
        for i in range(1, len(series_ind)):
            if series_ind[i] != series_ind[i-1] + 1:  # break in continuity
                if len(series_ind[start:i]) > self.lags_r:
                    series_segments.append(series_ind[start:i])
                    time_index.append(series_ind[start:i-self.lags_r])
                start = i
        if len(series_ind[start:]) > self.lags_r:
            series_segments.append(series_ind[start:])
            time_index.append(series_ind[start:-self.lags_r])
        y_out, y_target = self.TV_AR_lagged_one(series[series_segments[0]].reshape(-1,1))
        time_out = time_index[0]
        for i in range(1, len(series_segments)):
            segment = series_segments[i]
            y_out_temp, y_target_temp = self.TV_AR_lagged_one(series[segment].reshape(-1,1))
            y_out = np.concatenate([y_out, y_out_temp], axis=0)
            y_target = np.concatenate([y_target, y_target_temp], axis=0)
            time_out = np.concatenate([time_out, time_index[i]])
        return y_out, y_target, time_out.reshape(-1,1) + self.lags_r
    
    def recursion_pred(self, model, pred_step, t_start, y_start):
        pred_res = []
        for t in range(pred_step):
            g = tf.Variable(t_start + t, dtype = 'float64', trainable = False)
            g = tf.reshape(g, (-1,1))
            coef = np.array(model(g)).reshape(-1,)
            pred_one = np.sum(coef * y_start)
            pred_res.append(pred_one)
            y_start = np.concatenate([[1], y_start[2:], [pred_one]])
            t = t+1
        return pred_res

    def windows_one_pred(self,model, y_out, time_out, pred_dict, pred_step):
        for i in range(len(time_out)):
            t_start = time_out[i]
            y_start = y_out[i]
            rec_res = self.recursion_pred(model, pred_step, t_start, y_start)
            for k in range(pred_step):
                pred_dict[t_start + k].append(rec_res[k])
        return pred_dict

    def reconstruction_main(self, model, orig_sig, pred_step, lags_r, windows_leng):
        sig_dict = defaultdict(list)
        for i in tqdm(range(len(orig_sig)-windows_leng)):
            sig_windows = orig_sig[i:i+windows_leng]
            # print(sig_windows)
            temp_windows = self.TV_AR_lagged_main(sig_windows)
            
            sig_dict = self.windows_one_pred(model, temp_windows[0], temp_windows[2].reshape(-1,)+i, sig_dict, pred_step)
        return sig_dict
    
    def predict_step(self, signal_missing):
        max_p = 0
        max_seg = []
        for i in signal_missing:
            if i == True:
                max_seg.append(1)
            else:
                temp = len(max_seg)
                max_seg = []
                if temp > max_p:
                    max_p = temp
        temp = len(max_seg)
        if temp > max_p:
            max_p = temp
        return max_p
    
    def main(self):
        p = self.predict_step(np.isnan(self.obs_signal)) 
        signal_init = pd.Series(self.obs_signal).interpolate().to_numpy()
        p = 1
        d = self.reconstruction_main(self.model, signal_init, p, self.lags_r, self.win_leng)
        for i in range(len(signal_init)):
            d[i].append(signal_init[i])

        sing_out_temp = []
        for i in range(len(signal_init)):
            temp = np.array(d[i])
            temp = temp[~np.isnan(temp)]
            sigma = np.sqrt(np.var(temp))
            temp_wind = temp[temp<sigma*self.sigma_w]
            if len(temp_wind) == 0:
                sing_out_temp.append(np.median(temp))
            else:
                sing_out_temp.append(np.median(temp_wind))
        return np.array(sing_out_temp)
