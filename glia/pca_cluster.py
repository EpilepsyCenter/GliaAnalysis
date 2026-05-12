"""PCA + k-means with elbow / silhouette guidance."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def fit_pca(df: pd.DataFrame, feature_cols: list[str], n_components: int = 5):
    X = StandardScaler().fit_transform(df[feature_cols].to_numpy())
    pca = PCA(n_components=n_components)
    Y = pca.fit_transform(X)
    cols = [f"PC{i+1}" for i in range(n_components)]
    pc_df = pd.DataFrame(Y, columns=cols, index=df.index)
    return pca, pd.concat([pc_df, df], axis=1)


def kmeans_cluster(
    df: pd.DataFrame, feature_cols: list[str], k: int, random_state: int = 0
) -> pd.Series:
    X = StandardScaler().fit_transform(df[feature_cols].to_numpy())
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    return pd.Series(km.fit_predict(X), index=df.index, name="Cluster")


def cluster_selection_scan(
    df: pd.DataFrame, feature_cols: list[str], k_range=range(2, 9)
) -> pd.DataFrame:
    """Returns inertia (elbow) + silhouette score across k values."""
    X = StandardScaler().fit_transform(df[feature_cols].to_numpy())
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        sil = silhouette_score(X, km.labels_) if k > 1 else np.nan
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)
