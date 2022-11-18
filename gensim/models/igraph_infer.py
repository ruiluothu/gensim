import igraph as ig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
# from gensim.test.utils import common_corpus, common_dictionary
# from gensim.models import HdpModel
from collections import Counter, defaultdict
from hdpmodel_ADND import HdpModel
# from hdpmodel import HdpModel

n = 40
k = 3
block_sizes = np.ceil((n-k*10) * np.diff(np.sort(np.random.rand(k - 1)), prepend=[0])) + 10
block_sizes = np.hstack([block_sizes, np.array([n - sum(block_sizes)])]).tolist()
print(block_sizes)
A = (0.3 + 0.4 * np.random.rand(k, k)) * ((k/n/5) * np.ones((k, k)) + (8*k/n) * np.eye(k))
pref_matrix = ((A + A.T)/2).tolist()
G = ig.Graph.SBM(n, pref_matrix, block_sizes, directed=True, loops=False)
node_labels = reduce(lambda x, y: x + y, [[i] * int(size) for i, size in enumerate(block_sizes)])

edge_list = np.vstack([np.repeat(np.array([e.source_vertex.index, e.target_vertex.index]).reshape(1, -1), np.random.poisson(10) + 1, axis=0)  # np.random.poisson(3)
                if node_labels[e.source_vertex.index] == node_labels[e.target_vertex.index]
                else np.repeat(np.array([e.source_vertex.index, e.target_vertex.index]).reshape(1, -1), np.random.poisson(1) + 1, axis=0)  for e in G.es]).tolist()
dictionary = {key: key for key in set(np.array(edge_list).flatten())}
edge_counter = Counter(tuple(map(tuple, edge_list)))
node_counter = Counter(np.array(edge_list).flatten())
corpus_u = [list(Counter(np.array(edge_list)[:, 0].flatten()).items())]
corpus_v = [list(Counter(np.array(edge_list)[:, 1].flatten()).items())]

# hdp = HdpModel(corpus_u, dictionary, alpha=1, gamma=1, K=100, T=200, var_converge=1e-20)

hdp = HdpModel(None, dictionary, alpha=.1, gamma=.1, K=15, T=150, var_converge=1e-10)
hdp.update(corpus_u, sender=True)
hdp.update(corpus_v, sender=False)
for i in list(dictionary.keys()) + ['0x', 'happy', 120]:
    if np.random.rand() < 0.5:
        print(f"sender {i}: {hdp.inference_new_word(i, sender=True)}")
    else:
        print(f"receiver {i}: {hdp.inference_new_word(i, sender=False)}")

r"""
Implement the variational inference algorithm:
(1) initialize the variational parameters, which include: the corpus-level stick weights $\beta_i^{(H)}$, the document-level stick weights $\beta_t^{(A)}$ and $\beta_t^{(B)}$, the truncated (to $K^{(H)}$) corpus-level topic distribution, the truncated (to $K^{(A)}$) document-level topic distribution. Note that the initialization for $lambda \in {\mathbb{R}^{+}}^{num_truncations * num_words} is such that more words and larger truncations will lead to smaller initialized value, i.e., weaker Dirichlet prior 

self.m_Elogbeta represents the E[log(beta)] such that beta~Dir(\lambda)


(2) update the parameters with the training documents

(3) do the inference: for both levels, the likelihood is the dot product of the stick proportion and the per-stick dirichlet distribution
"""


