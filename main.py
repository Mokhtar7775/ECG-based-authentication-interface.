import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import wfdb as wfdb


def extract_r_peaks(sig, sig1, win_size):
    ls = []
    r_peaks_temp = []
    on = []
    off = []
    for i in range(0, len(sig)):
        if i < win_size:
            ls.append(sig[i])
        else:
            win_size += 110
            max_value = max(ls)
            if max_value > (0.75 * np.max(sig)):
                win_start = i - 110
                max_indx_win = ls.index(max_value)
                max_original = win_start + max_indx_win
                r_peaks_temp.append(max_original)
            ls = []
    r_peaks = []
    i = 0
    while i < (len(r_peaks_temp) - 1):
        if r_peaks_temp[i + 1] - r_peaks_temp[i] < 110:
            v1 = ecg_original_signal[r_peaks_temp[i]]
            v2 = ecg_original_signal[r_peaks_temp[i + 1]]
            if v1 > v2:
                r_peaks.append(r_peaks_temp[i])
            else:
                r_peaks.append(r_peaks_temp[i + 1])
            i = i + 1
        else:
            r_peaks.append(r_peaks_temp[i])
        i = i + 1

    for j in range(len(r_peaks)):
        max_curve_idx_on = min_radius_of_curvature(sig1, r_peaks[j] - 55, r_peaks[j] - 40)
        onset = max_curve_idx_on + (r_peaks[j] - 55)
        max_curve_idx_off = min_radius_of_curvature(sig1, r_peaks[j] + 40, r_peaks[j] + 55)
        offset = max_curve_idx_off + r_peaks[j] + 40
        on.append(onset)
        off.append(offset)
    return r_peaks, on, off


def q_s(s, r, on, off):
    qq = []
    ss = []
    for i in range(0, len(r)):
        qq.append(np.argmin(s[on[i]:r[i]]) + on[i])
        ss.append(np.argmin(s[r[i]:off[i]]) + r[i])
    return qq, ss


def p_on_off(sig, q_on, r):
    win_size = 200
    p_peak = []
    b_p, a_p = signal.butter(2, [1.0 / nyq, 10.0 / nyq], btype='bandpass')
    p_on = []
    p_off = []
    for i in range(1, len(r)):
        filtered_for_p = signal.filtfilt(b_p, a_p, sig[q_on[i] - win_size:q_on[i]])
        p_peak.append(np.argmax(filtered_for_p) + (q_on[i] - win_size))
        slice = sig[p_peak[i-1]:q_on[i]-10]
        slice_filtered = signal.filtfilt(b_p,a_p,slice)
        max_curve_idx_on = min_radius_of_curvature(sig, p_peak[i - 1] - 100, p_peak[i - 1])
        onset = max_curve_idx_on + p_peak[i - 1] - 100
        max_curve_idx_off = min_radius_of_curvature(slice_filtered, 0, len(slice_filtered))
        offset = max_curve_idx_off + p_peak[i - 1]
        p_on.append(onset)
        p_off.append(offset)
    return p_peak,p_on,p_off



def min_radius_of_curvature(sig, n1, n2):
    x_axis = np.arange(n1, n2)
    y_axis = sig[n1:n2]
    poly = np.polyfit(x_axis, y_axis, deg=3)
    d1_poly = np.polyder(poly)
    d2_poly = np.polyder(d1_poly)
    curvature_vals = []
    for t in x_axis:
        poly_val = np.polyval(poly, t)
        poly1d_val = np.polyval(d1_poly, t)
        poly2d_val = np.polyval(d2_poly, t)
        curvature_val = np.linalg.norm((poly1d_val * poly2d_val)) / np.power(np.linalg.norm(poly1d_val), 3)
        curvature_vals.append(curvature_val)
    curvature_vals = np.array(curvature_vals)
    max_curve_idx = np.argmax(curvature_vals)
    return max_curve_idx


def t_on_off(sig, q_off, r):
    window_size = 350
    t_peak = []
    t_on = []
    t_off = []
    b_t, a_t = signal.butter(2, [0.5 / nyq, 8.0 / nyq], btype='bandpass')
    for i in range(1, len(r)-1):
        filtered_for_t = signal.filtfilt(b_t, a_t, sig[q_off[i]:q_off[i] + window_size])
        t_peak.append(np.argmax(filtered_for_t) + q_off[i])
        max_curve_idx_on = min_radius_of_curvature(sig, t_peak[i-1]-100, t_peak[i - 1])
        onset = max_curve_idx_on + t_peak[i-1]-100
        max_curve_idx_off = min_radius_of_curvature(sig, t_peak[i - 1], t_peak[i-1]+100)
        offset = max_curve_idx_off + t_peak[i - 1]
        t_on.append(onset)
        t_off.append(offset)
    return t_peak, t_on, t_off


dataset = os.listdir('DataSet')
for patient in dataset:
    records = os.listdir('DataSet\\' + patient)
    for record in records:
        record1 = wfdb.rdrecord('Dataset' + '/' + patient + '/' + record + "/" + record, channels=[11])
        ecg_original_signal = record1.p_signal.flatten()
        fs = record1.fs
        nyq = 0.5 * fs
        f_low = 1.0 / nyq
        f_high = 40.0 / nyq
        b, a = signal.butter(2, [f_low, f_high], btype="bandpass")
        preprocessed = signal.filtfilt(b, a, ecg_original_signal)

        # low,high
        b_l, a_l = signal.butter(4, 40.0 / nyq, btype='lowpass')
        low_filtered = signal.filtfilt(b_l, a_l, preprocessed)

        b_h, a_h = signal.butter(4, 10.0 / nyq, btype='highpass')
        high_filtered = signal.filtfilt(b_h, a_h, low_filtered)

        differentiated_signal = np.diff(high_filtered, n=1)
        squared_signal = differentiated_signal ** 2
        window_size = 110
        window = np.ones(window_size) / window_size
        moving_avg = np.convolve(squared_signal, window, mode='same')
        b_l, a_l = signal.butter(4, 30.0 / nyq, btype='lowpass')
        mo = signal.filtfilt(b_l, a_l, moving_avg)
        r_peak, qrs_on, qrs_off = extract_r_peaks(moving_avg, mo, window_size)

        q, s = q_s(high_filtered, r_peak, qrs_on, qrs_off)
        p_peaks,p_on,p_off = p_on_off(preprocessed,qrs_on,r_peak)
        t_peaks, t_on, t_off = t_on_off(preprocessed, qrs_off, r_peak)
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        axs[0].plot(np.arange(0, len(ecg_original_signal)), ecg_original_signal)
        axs[0].set_title("Original")
        axs[0].scatter(qrs_on, ecg_original_signal[qrs_on], marker='o', color='yellow')
        axs[0].scatter(qrs_off, ecg_original_signal[qrs_off], marker='o', color='green')
        axs[0].scatter(r_peak, ecg_original_signal[r_peak], marker='x', color='red')
        axs[0].scatter(q, ecg_original_signal[q], marker='o', color='purple')
        axs[0].scatter(s, ecg_original_signal[s], marker='o', color='purple')
        axs[0].scatter(p_on, ecg_original_signal[p_on], marker='o', color='black')
        axs[0].scatter(p_off, ecg_original_signal[p_off], marker='o', color='black')
        axs[0].scatter(p_peaks, ecg_original_signal[p_peaks], marker='o', color='red')
        axs[0].scatter(t_peaks, ecg_original_signal[t_peaks], marker='o', color='orange')
        axs[0].scatter(t_on, ecg_original_signal[t_on], marker='o', color='orange')
        axs[0].scatter(t_off, ecg_original_signal[t_off], marker='o', color='orange')
        axs[1].plot(np.arange(0, len(mo)), mo)
        plt.show()