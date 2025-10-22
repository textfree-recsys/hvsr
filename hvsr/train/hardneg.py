import numpy as np
from sklearn.neighbors import NearestNeighbors

def fit_knn(emb, k=20, metric='cosine', n_jobs=-1):
    nn = NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=n_jobs)
    nn.fit(emb)
    d, nbrs = nn.kneighbors(emb, return_distance=True)
    return nbrs[:, 1:1+k].astype(np.int32)
