import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import glob
import librosa

dataset = glob.glob('DataSet/*')
for patient in dataset:
        records = glob.glob(patient+'\\*')
        for record in records:
                file = np.fromfile(record,dtype='byte')
                nyq = 0.5 * 500
                low = 1.0 / nyq
                high = 40.0 / nyq
                num, den = signal.butter(2, [low, high], btype='bandpass', output='ba')
                plt.plot(np.arange(0, 100), file[slice(100)])
                plt.title(" ECG Signal of "+record[11:])
                plt.show()
                amp = list(file)

                filtered = signal.lfilter(num, den, amp)
                plt.plot(np.arange(0, 100), filtered[slice(100)])
                plt.title("Filtered Signal of "+record[11:])
                plt.show()
                print(len(filtered))
                diff = []
                for y in range(0, len(filtered) - 1):
                        yDiff = (filtered[y + 1] - filtered[y])
                        diff.append(yDiff)
                print(len(diff))
                plt.plot(np.arange(0, 100), diff[slice(100)])
                plt.title("Derevatived Signal of "+record[11:])
                plt.show()
                sq = []
                print(len(sq))
                plt.plot(np.arange(0, 100), sq[slice(100)])
                plt.title("Squaring Signal of "+record[11:])
                plt.show()
                # sq = np.array(sq)
                frames = librosa.util.frame(sq,frame_length= len(sq),hop_length= 1)
                print("frames : ",frames)
                windowed_frames = np.hanning(len(sq)).reshape(-1,1)*frames
                print("windowed : ",windowed_frames)
                plt.plot(np.arange(0,100),windowed_frames[slice(100)])
                plt.title("Windowed Signal of "+record[11:])
                plt.show()



