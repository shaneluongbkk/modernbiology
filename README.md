# Deep Learning - Modern Biology Study Group

## Input

### Features (X)
- A NumPy array of shape `(n_samples, n_features)` representing the feature space.

### True Labels (y_true)
- A NumPy array of shape `(n_samples,)` containing the ground truth cluster labels for each data point.
- Used to calculate Adjusted Rand Index and Normalized Mutual Information.

## Output

### Visualization
- Scatter plots of the dataset with true labels and predicted clusters.
- A contingency matrix heatmap displaying the relationship between true and predicted labels.

### Evaluation Metrics

#### Silhouette Score
The Silhouette Score is calculated as follows:

$$
s(i) = \frac{b(i) - a(i)}{\max (\{a(i), b(i))\}}
$$

Where:
- \(a(i)\): The average distance from \(i\) to all data points in the same cluster \(c_p\).
- \(b(i)\): The lowest average distance from \(i\) to all data points in the same cluster \(c\) among all clusters \(c\).

Clusters can be replaced with batches if one is estimating the silhouette width to assess batch effects.

---

#### Adjusted Rand Index (ARI)
The Adjusted Rand Index is calculated as:

$$
\text{ARI} = \frac{\sum_{ij} \binom{n_{ij}}{2} - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right] \binom{n}{2}}{\frac{1}{2} \left[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\right] - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right] \binom{n}{2}}
$$

Where:
- \(n_{ij}\), \(a_i\), and \(b_j\) are values from the contingency table.

---

#### Normalized Mutual Information (NMI)
The Normalized Mutual Information is calculated as:

$$
\text{NMI} = \frac{I(P; T)}{\sqrt{H(P) H(T)}}
$$

Where:
- \(P, T\): Empirical categorical distributions for the predicted and real clustering.
- \(I\): The mutual entropy.
- \(H\): The Shannon entropy.



