import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import dataclass
from typing import Dict, List, Literal

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from preprocessing.data_preprocessing import Config

# Use TkAgg backend for matplotlib
matplotlib.use("TkAgg")



# ============================================================
# 1. Data Profiler
# ============================================================

class DataProfiler:
    """Profile data segments and visualize them via heatmaps."""

    def __init__(self, client_groups: Dict[str, List], original_df: pd.DataFrame):
        self.client_groups = client_groups
        self.client_groups['all_clients'] = original_df['customer_id'].tolist()
        self.metrics = Config().engineered_features
        self.original_df = original_df

        self.metrics.update({
        "mean_age": lambda df: df["age"]
    })

    def profile_segment(self, segment: pd.DataFrame) -> Dict[str, float]:
        """Compute mean metrics for a segment."""
        return {name: func(segment).mean() for name, func in self.metrics.items()}

    def profile_all_segments(self) -> pd.DataFrame:
        """Profile all client segments and return a DataFrame."""
        segment_profiles = {
            name: self.profile_segment(self.original_df[self.original_df["customer_id"].isin(ids)])
            for name, ids in self.client_groups.items()
        }
        return pd.DataFrame(segment_profiles).T
    
    def select_informative_features(self, profile_df: pd.DataFrame, top_n: int = 10) -> List[str]:
        """
        Select top N features with the largest variance across segments.
        These features are most likely to show differences in the heatmap.
        """
        variances = profile_df.var(axis=0)
        top_features = variances.sort_values(ascending=False).head(top_n).index.tolist()
        print(top_features)
        return top_features

    def plot_heatmap(self, profile_df: pd.DataFrame, figsize=(12, 8), cmap="RdYlGn"):
        """
        Plot a heatmap of segment profiles with column-wise normalization.
        Columns with identical values are rendered in neutral color.
        """
        # Copy df to avoid modifying original
        normed_df = profile_df.copy().astype(float)

        # Column-wise normalization
        for col in normed_df.columns:
            min_val, max_val = normed_df[col].min(), normed_df[col].max()
            if min_val == max_val:
                # Constant column → set to NaN (renders neutral)
                normed_df[col] = np.nan
            else:
                normed_df[col] = (normed_df[col] - min_val) / (max_val - min_val)

        plt.figure(figsize=figsize)
        sns.heatmap(
            normed_df,
            annot=profile_df,  # show actual values
            fmt=".2f",
            cmap=cmap,
            linewidths=0.5,
            linecolor="gray",
            cbar_kws={"label": "Normalized Value"},
            center=0.5,  # optional: center the colormap
        )

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.title("Segment Profiles Heatmap (Column-normalized)", fontsize=16)
        plt.tight_layout()
        plt.show(block=True)

    def profile_and_plot(self) -> pd.DataFrame:
        """Profile all segments and plot the heatmap."""
        profile_df = self.profile_all_segments()
        self.plot_heatmap(profile_df)
        return profile_df


# ============================================================
# 2. Clustering Model Wrapper
# ============================================================

@dataclass
class ClusterConfig:
    algorithm: Literal["kmeans", "dbscan"] = "kmeans"
    n_clusters: int = 5
    eps: float = 0.5
    min_samples: int = 10
    random_state: int = 42


class ClusterModelWrapper:
    """Unified wrapper for KMeans and DBSCAN clustering algorithms."""

    def __init__(self, config: ClusterConfig):
        self.config = config
        self.model = self._init_model()

    def _init_model(self):
        if self.config.algorithm == "kmeans":
            return KMeans(n_clusters=self.config.n_clusters, random_state=self.config.random_state)
        elif self.config.algorithm == "dbscan":
            return DBSCAN(eps=self.config.eps, min_samples=self.config.min_samples)
        else:
            raise ValueError("Unsupported algorithm. Use 'kmeans' or 'dbscan'.")

    def fit(self, X: pd.DataFrame):
        self.model.fit(X)
        return self

    def predict(self, X: pd.DataFrame):
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        # DBSCAN does not have predict
        return self.model.fit_predict(X)

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)


class Clusterer:
    """Cluster data and compute clustering metrics."""

    def __init__(self, cluster_model: ClusterModelWrapper, df: pd.DataFrame):
        self.cluster_model = cluster_model
        self.df = df

    def fit_and_predict(self) -> pd.DataFrame:
        """Fit the model and assign cluster labels."""
        X = self.df.drop(columns=["customer_id"])
        self.cluster_model.fit(X)
        labels = self.cluster_model.predict(X)

        df_clusters = self.df[["customer_id"]].copy()
        df_clusters["cluster"] = labels
        return df_clusters

    def display_metrics(self, df_clusters: pd.DataFrame):
        """Display clustering metrics if more than 1 cluster exists."""
        X = self.df.drop(columns=["customer_id"])
        labels = df_clusters["cluster"].values

        unique_labels = set(labels)
        if len(unique_labels - {-1}) > 1:  # ignore noise for DBSCAN
            print(f"Silhouette Score: {silhouette_score(X, labels):.3f}")
            print(f"Calinski-Harabasz Index: {calinski_harabasz_score(X, labels):.3f}")
            print(f"Davies-Bouldin Index: {davies_bouldin_score(X, labels):.3f}")
        else:
            print("Not enough clusters to compute metrics or only noise points (-1) present.")


# ============================================================
# 3. Main
# ============================================================

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/customer_churn_synthetic.csv")
    df_preprocessed = pd.read_csv("data/customer_churn_synthetic_preprocessed_with_pca.csv")

    # Configure and run clustering
    config = ClusterConfig(algorithm="kmeans", n_clusters=3, random_state=42)
    cluster_model = ClusterModelWrapper(config=config)
    clusterer = Clusterer(cluster_model=cluster_model, df=df_preprocessed)

    df_clusters = clusterer.fit_and_predict()
    clusterer.display_metrics(df_clusters=df_clusters)

    # Create segment dictionary for profiling
    cluster_segments = {
        str(c): df_clusters[df_clusters["cluster"] == c]["customer_id"].tolist()
        for c in df_clusters["cluster"].unique()
    }

    # Profile segments
    profiler = DataProfiler(client_groups=cluster_segments, original_df=df)
    profile_df = profiler.profile_and_plot()


    #profile churn
    churn_segments = {'churned': df[df['churn'] == 1]['customer_id'].tolist(),                     
                      'not_churned': df[df['churn'] == 0]['customer_id'].tolist()}
    profiler = DataProfiler(client_groups=churn_segments, original_df=df)
    profile_df = profiler.profile_and_plot()