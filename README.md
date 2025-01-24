# Deep Learning - Modern Biology Study Group
Input

The input consists of a dataset that needs clustering and evaluation. Specifically:

Features (X):

A NumPy array of shape (n_samples, n_features) representing the feature space.

True Labels (y_true):

A NumPy array of shape (n_samples,) containing the ground truth cluster labels for each data point.

Used to calculate Adjusted Rand Index and Normalized Mutual Information.

Output

The script produces the following outputs:

Visualization:

Scatter plots of the dataset with true labels and predicted clusters.

A contingency matrix heatmap displaying the relationship between true and predicted labels.

Evaluation Metrics:

Silhouette Score: Measures how similar a data point is to its cluster compared to other clusters. Output as a float.

Adjusted Rand Index (ARI): Evaluates the similarity between true and predicted labels, adjusted for chance. Output as a float.

Normalized Mutual Information (NMI): Measures the amount of information shared between true and predicted labels. Output as a float.
