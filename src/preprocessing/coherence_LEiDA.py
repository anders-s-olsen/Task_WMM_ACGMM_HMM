import numpy as np

def coherenceMap(theta_signals):
    N, T = theta_signals.shape

    all_coh_maps = np.zeros((T, N, N))

    for t in range(T):
        signals_t = theta_signals[:, t]
        coh_map_t = np.cos(signals_t[:, None] - signals_t[None, :])
        all_coh_maps[t] = coh_map_t

    return all_coh_maps


def leadingEigenVec(subject_coherence_maps):
    EigValues, EigVectors = np.linalg.eigh(subject_coherence_maps)
    # dims: Values=(330, 90), Vectors=(330, 90, 90)
    # The eigen values is sorted in ascending order, we take index [-1]

    LEiDA_vectors = EigVectors[:, :, -1]

    return LEiDA_vectors

