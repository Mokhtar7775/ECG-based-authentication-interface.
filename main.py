import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import glob
import librosa


def moving_window_integration(signal):
        window_size = int(0.1* 1000)  # Window size for integration
        signal_int = np.convolve(signal, np.ones(window_size), mode='valid')

        # Thresholding
        # threshold = 0.4 * np.max(signal_int)  # Threshold for QRS detection
        # qrs_detect = np.zeros(signal_int.shape)
        # qrs_detect[signal_int > threshold] = 1
        return signal_int
dataset = glob.glob('DataSet/*')
for patient in dataset:
        records = glob.glob(patient+'\\*')
        for record in records:
                file = np.fromfile(record,dtype='byte')
                nyq = 0.5 * 500
                low = 1.0 / nyq
                high = 40.0 / nyq
                num, den = signal.butter(2, [low, high], btype='bandpass', output='ba')
                plt.plot(np.arange(0, 200), file[slice(200)])
                plt.title(" ECG Signal of "+record[11:])
                plt.show()
                amp = list(file)

                filtered = signal.lfilter(num, den, amp)
                plt.plot(np.arange(0, 200), filtered[slice(200)])
                plt.title("Filtered Signal of "+record[11:])
                plt.show()
                print(len(filtered))
                diff = np.diff(filtered,n=1)
                print(len(diff))
                plt.plot(np.arange(0, 200), diff[slice(200)])
                plt.title("Derevatived Signal of "+record[11:])
                plt.show()
                sq = diff**2
                print(len(sq))
                plt.plot(np.arange(0, 200), sq[slice(200)])
                plt.title("Squaring Signal of "+record[11:])
                plt.show()
                win = moving_window_integration(sq)
                plt.plot(np.arange(0, 200), win[slice(200)])
                plt.title("Windowed Signal of " + record[11:])
                plt.show()



