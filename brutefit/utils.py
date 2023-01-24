import numpy as np

def build_desmat(X, poly_max=1, include_bias=True, max_interaction_order=0):
    """
    Build design matrix to cover all model permutations
    """
    if include_bias:
        desmat = [np.ones(X.shape[0]).reshape(-1, 1)]
    else:
        desmat = []
    
    for o in range(1, poly_max + 1):
        desmat.append(X**o)

    interaction_pairs = np.vstack(np.triu_indices(X.shape[-1], 1)).T

    for o in range(1, max_interaction_order + 1):
        for ip in interaction_pairs:
            desmat.append((X[:, ip[0]]**o * X[:, ip[1]]**o).reshape(-1, 1))
    return np.hstack(desmat)