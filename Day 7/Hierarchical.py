
#! Implement a hierarchical clustering

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def main() -> None:
    iris = load_iris()
    x = iris.data
    y_true = iris.target

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    z = linkage(x_scaled, method="ward")

    model = AgglomerativeClustering(
        n_clusters=3,
        metric="euclidean",
        linkage="ward",
    )
    y_pred = model.fit_predict(x_scaled)

    sil_score = silhouette_score(x_scaled, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    print("Hierarchical Clustering on Iris Dataset")
    print(f"Number of clusters: {len(np.unique(y_pred))}")
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Adjusted Rand Index (vs true labels): {ari:.4f}")
    print(f"Cluster labels (first 20): {y_pred[:20]}")

    plt.figure(figsize=(10, 6))
    dendrogram(z, truncate_mode="lastp", p=20, leaf_rotation=45, leaf_font_size=10)
    plt.title("Hierarchical Clustering Dendrogram (Ward)")
    plt.xlabel("Cluster or Sample Index")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


