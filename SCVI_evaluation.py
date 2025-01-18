import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from collections import Counter
from math import comb, log2

X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=2.5, random_state=42)

# Display the sample data
plt.scatter(X[:, 0], X[:, 1], c=y_true, s=50, cmap='viridis')
plt.title("Sample Data with True Labels (Overlapping Clusters)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Perform clustering with KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(X)

# Display the clustering results
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X')
plt.title("Clustering Results with KMeans (Overlapping Clusters)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Silhouette Score
def silhouette_score_manual(X, labels):
    n = len(X)
    silhouette_scores = []
    for i in range(n):
        same_cluster = X[labels == labels[i]]
        other_clusters = X[labels != labels[i]]
        a = np.mean([np.linalg.norm(X[i] - point) for point in same_cluster])
        b = np.min([np.mean([np.linalg.norm(X[i] - point) for point in X[labels == label]]) for label in set(labels) if label != labels[i]])
        silhouette_scores.append((b - a) / max(a, b))
    return np.mean(silhouette_scores)

sil_score = silhouette_score_manual(X, y_pred)
print(f"Silhouette Score (manual): {sil_score:.4f}")

# Adjusted Rand Index (ARI)
def adjusted_rand_index_manual(true_labels, pred_labels):
    contingency = pd.crosstab(pd.Series(true_labels), pd.Series(pred_labels))
    sum_comb_c = sum(comb(n_ij, 2) for n_ij in contingency.to_numpy().flatten())
    sum_comb_rows = sum(comb(n_i, 2) for n_i in contingency.sum(axis=1))
    sum_comb_cols = sum(comb(n_j, 2) for n_j in contingency.sum(axis=0))
    n = len(true_labels)
    expected_index = (sum_comb_rows * sum_comb_cols) / comb(n, 2)
    max_index = (sum_comb_rows + sum_comb_cols) / 2
    return (sum_comb_c - expected_index) / (max_index - expected_index)

ari_score = adjusted_rand_index_manual(y_true, y_pred)
print(f"Adjusted Rand Index (ARI) (manual): {ari_score:.4f}")

# Normalized Mutual Information (NMI)
def normalized_mutual_info_manual(true_labels, pred_labels):
    contingency = pd.crosstab(pd.Series(true_labels), pd.Series(pred_labels)).to_numpy()
    n = np.sum(contingency)
    pi = np.sum(contingency, axis=1)
    pj = np.sum(contingency, axis=0)
    mutual_info = 0
    for i in range(len(pi)):
        for j in range(len(pj)):
            if contingency[i][j] > 0:
                mutual_info += contingency[i][j] / n * log2((contingency[i][j] * n) / (pi[i] * pj[j]))
    entropy_true = -sum((pi / n) * np.log2(pi / n))
    entropy_pred = -sum((pj / n) * np.log2(pj / n))
    return mutual_info / np.sqrt(entropy_true * entropy_pred)

nmi_score = normalized_mutual_info_manual(y_true, y_pred)
print(f"Normalized Mutual Information (NMI) (manual): {nmi_score:.4f}")

# Contingency Matrix
conf_matrix = pd.crosstab(pd.Series(y_true), pd.Series(y_pred))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Contingency Matrix Between True Labels and Predicted Labels")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
