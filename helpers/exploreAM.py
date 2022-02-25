# EXPERIMENT: EXPLORE AM DATASET
import os.path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def compute_mds(matrix_corr):
    D = 1-matrix_corr
    # Number of points
    n = len(D)

    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n

    # YY^T
    B = -H.dot(D ** 2).dot(H) / 2

    # Diagonalize
    evals, evecs = np.linalg.eigh(B)

    # Sort by eigenvalue in descending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    L = np.diag(np.sqrt(evals[w]))
    V = evecs[:, w]
    Y = V.dot(L)

    return Y, evals


# LOAD AM DATA WITH DOMAIN-SPECIFIC SENTIMENT SCORES [TRAIN DATA]
domain_path = '../data/multi-domain-am/domain_list'
am_embedding_path = '../data/multi-domain-am/embedding_am.txt'
am_freq_path = '../data/multi-domain-am/embedding_am_freq_pd.txt'

domain_list = []
with open(domain_path, 'r') as f:
    for line in f:
        domain_list.append(line[:-1])
    domain_list[-1] = line

nTokens = int(domain_list[0][1:7])
nDomains = int(domain_list[0][9:11])
domain_list = domain_list[1:]

sentiment_df = pd.read_csv('../outputs/predicted_sentiments_glove.csv')

# COMPUTE CORRELATION COEFFICIENTS BETWEEN PREDICTED SENTIMENT SCORES FROM DIFFERENT DOMAINS
scores = np.empty((nDomains, len(sentiment_df)), dtype=np.float32)
for domain_index in range(nDomains):
    scores[domain_index] = sentiment_df[domain_list[domain_index]].tolist()

cor_matrix = np.corrcoef(scores)

cor_matrix_df = pd.DataFrame(index=domain_list)
for domain_index in range(nDomains):
    cor_matrix_df[domain_list[domain_index]] = cor_matrix[domain_index, :]

reduced_space, axis_weights = compute_mds(cor_matrix)

sns.set()
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
heatmap = sns.heatmap(cor_matrix_df, annot=False, fmt='0.2f', xticklabels=1, annot_kws={'rotation': 90, 'size': 8})
plt.title('(a) Correlation heatmap', fontsize=16)

heatmap.set_xticklabels(
    heatmap.get_xticklabels(),
    rotation=50,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-small'
)

heatmap.set_yticklabels(
    heatmap.get_yticklabels(),
    rotation=50,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-small'
)


ax = fig.add_subplot(122)
ax.scatter(reduced_space[:, 0], reduced_space[:, 1])
plt.title('(b) Sentiment-based cross-domain relations in 2D plane', fontsize=16)

text_offsets = [[-0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.25, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03],
                [-0.017, -0.017, 0.007, 0.007, 0.007, 0.007, -0.017, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, -0.017, 0.007, 0.007, 0.007, -0.017, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007]]
for i, txt in enumerate(domain_list):
    ax.annotate(txt, (reduced_space[i, 0] + text_offsets[0][i], reduced_space[i, 1] + text_offsets[1][i]))
# plt.plot(reduced_space[:, 0], reduced_space[:, 1], 'r*')
plt.tight_layout()
plt.show()

print('End!')