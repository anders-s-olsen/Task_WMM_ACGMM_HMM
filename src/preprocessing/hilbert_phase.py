import numpy as np
from scipy.signal import hilbert

# return the real and analytical signal of a Real signing
def hilbert_transform(signal):
    A_t = hilbert(signal) # outputs complex values
    s_t , Hs_t = np.real(A_t),np.imag(A_t) # seperate real and imaginary
    return s_t, Hs_t

# Complete Hilbert tranformation and phase exstraction
def hilbert_phase_extract(signal):
    return np.angle(hilbert(signal))
