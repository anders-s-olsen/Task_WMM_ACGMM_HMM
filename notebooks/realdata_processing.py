import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm, trange
from matplotlib import animation
from scipy.signal import hilbert
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import h5py

sys.path.append(os.path.abspath(os.path.join('..')))

from src.preprocessing.bandpass_filter import butter_bandpass_filter
from src.preprocessing.hilbert_phase import hilbert_phase_extract
from src.preprocessing.coherence_LEiDA import coherenceMap, leadingEigenVec

for file in list(glob.glob('../data/processed/*.h5')):
    data_subject = h5py.File(file, mode='r')
    data_tmp = torch.tensor(np.array(data_subject['data']))
    data = torch.transpose(data_tmp,dim0=0,dim1=1).float()

    