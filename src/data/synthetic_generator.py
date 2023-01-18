import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

@torch.no_grad()
def syntheticMixture3D(pi, Sigmas, num_points: int = 1000, point_dim: int = 3, as_array: bool = False):
    num_clusters = Sigmas.shape[0]
    print(f'Simulate {num_points} point from {num_clusters} of clusters')

    Lower_chol = torch.zeros(num_clusters, point_dim, point_dim)
    for idx, sig in enumerate(Sigmas):
        # sig = 100 * sig ACG is scale invariant
        Lower_chol[idx, :, :] = torch.linalg.cholesky(sig)

    # mixture assign and sample
    X = torch.zeros(num_points, point_dim)
    cluster_allocation = torch.zeros(num_points)
    for n in range(num_points):
        cluster_list = list(range(num_clusters))
        n_clust_id = int(np.random.choice(cluster_list, 1, p=pi))  # sample one cluster with pi probaility
        nx = Lower_chol[n_clust_id] @ torch.randn(point_dim)
        X[n] = nn.functional.normalize(nx, dim=0)
        cluster_allocation[n] = n_clust_id

    return (np.array(X), np.array(cluster_allocation)) if as_array else (X, cluster_allocation)

@torch.no_grad()
def syntheticHMM(pi, Sigmas, transition_matrix, seq_len: int = 150, num_subject = 20, point_dim: int = 3, as_array: bool = False):

    num_states = Sigmas.shape[0]
    print(f'Simulate sequence of length {seq_len} from {num_states} hidden states')

    Lower_chol = torch.zeros(num_states, point_dim, point_dim)

    for idx, sig in enumerate(Sigmas):
        # sig = 100 * sig ACG is scale invariant
        Lower_chol[idx, :, :] = torch.linalg.cholesky(sig)

    X_emission = torch.zeros(num_subject, seq_len, point_dim)
    Z_state_seq = torch.zeros(num_subject, seq_len)
    T_matrix = nn.functional.softmax(transition_matrix.to(torch.float64),dim=1)
    state_list = list(range(num_states))

    for sub in range(num_subject):

        for t in range(seq_len):
            if t == 0:
                t_state_id = int(np.random.choice(state_list, 1, p=pi))
            else:
                # get transition probs from state at time t-1 to all states at time t
                t_z_probs = T_matrix[int(Z_state_seq[sub, t - 1])] # row from transition matrix

                # get state for time t by weighting the transition probs
                #print(t_z_probs)
                t_state_id = int(np.random.choice(state_list, 1, p=t_z_probs))

            # Emission from state at time t.
            t_x = Lower_chol[t_state_id] @ torch.randn(point_dim)
            X_emission[sub, t] = nn.functional.normalize(t_x, dim=0)  # Projection to the sphere
            Z_state_seq[sub, t] = t_state_id  #keep track of seqence

    Z_state_seq = Z_state_seq.to(torch.int32)
    return (np.array(X_emission), np.array(Z_state_seq)) if as_array else (X_emission, Z_state_seq)


if __name__ == '__main__':
    sig1 = torch.diag(torch.tensor([1, 1e-3, 1e-3]))
    sig2 = torch.eye(3) + 0.9 * (torch.ones(3) - torch.eye(3))
    sig3 = torch.diag(torch.tensor([1e-3, 1, 1])) \
           + 0.9 * torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    # print(torch.linalg.eigh(sig1))
    SIGMAs = torch.stack([sig1, sig2, sig3], dim=0)
    #print(SIGMAs)
    PI = [0.5, 0.2, 0.3]
    X, cluster_id = syntheticMixture3D(pi=PI, Sigmas=SIGMAs, num_points=3000, as_array=True)
    print(X.shape)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    id_2_color = {0: 'cyan', 1: 'green', 2: 'magenta'}
    id_2_colorcode = {0: (1., 1., 1.), 1: (0.5, 0.5, 0.5), 2: (0., 0., 0.)}

    label_color = [id_2_color[id] for id in cluster_id]

    #ax.scatter(1, 1e-3, 1e-3, s=80, c='black')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=4, alpha=0.5, c=label_color)
    ax.set_xlabel('$x$', fontsize=15)
    ax.set_ylabel('$y$', fontsize=15)
    ax.set_zlabel('$z$', fontsize=15)
    plt.show()
