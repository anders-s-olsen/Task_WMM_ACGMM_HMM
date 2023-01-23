from scipy.fftpack import rfftfreq
from scipy.signal import butter, filtfilt

# Function for filtering a signal given low and high cut for bandpass filter with Frequencies
# Define filter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

# Apply Filter
# Changed by Anders to be correct range (from 0.01-0.1 to 0.009-0.08)
def butter_bandpass_filter(data, lowcut=0.009, highcut=0.08, fs=1/2.490, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


