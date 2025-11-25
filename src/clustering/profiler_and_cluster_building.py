import pandas as pd
import numpy as np
import joblib
from dataclasses import dataclass
from typing import Dict, Callable, List, Optional, Literal
import sys
import os

from sklearn.cluster import KMeans, DBSCAN
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg", depends on your system
import matplotlib.pyplot as plt

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.data_preprocessing import Config
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# ============================================================
#   DATA PROFILER
# ============================================================

class DataProfiler:
    
    def __init__(self, client_groups: Dict[str, List], original_df):
        self.client_groups = client_groups
        self.client_groups['all_clients'] = original_df['customer_id'].tolist()
        self.metrics =Config().engineered_features
        self.original_df = original_df
    
    def profile_segment(self, segment: pd.DataFrame) -> Dict[str, float]:
        profile_results = {}
        for metric_name, metric_func in self.metrics.items():
            profile_results[metric_name] = metric_func(segment).mean()
        return profile_results
    
    def profile_all_segments(self) -> pd.DataFrame:
        segment_profiles = {}
        for segment_name, client_ids in self.client_groups.items():
            segment_subset = self.original_df[self.original_df["customer_id"].isin(client_ids)]
            segment_profiles[segment_name] = self.profile_segment(segment_subset)
        return pd.DataFrame(segment_profiles).T
    
    def plot_heatmap(self, profile_df: pd.DataFrame, figsize=(12, 8), cmap="RdYlGn", filename=None):
        plt.figure(figsize=figsize)
        
        # Create the heatmap
        sns.heatmap(
            profile_df,
            annot=True,
            cmap=cmap,
            fmt=".2f",
            linewidths=0.5,
            linecolor="gray",
            cbar_kws={"label": "Mean Value"}
        )
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)  # keep y labels horizontal
        
        plt.title("Segment Profiles Heatmap", fontsize=16)
        plt.tight_layout()  # adjust layout to fit labels
        
        if filename:
            plt.savefig(filename, dpi=300)
            print(f"Saved heatmap to {filename}")
        else:
            plt.show(block=True)

    def profile_and_plot(self) -> pd.DataFrame:
        profile_df = self.profile_all_segments()
        self.plot_heatmap(profile_df)
        return profile_df


# ============================================================
#   CLUSTERING MODEL WRAPPER
# ============================================================

@dataclass
class ClusterConfig:
    algorithm: Literal["kmeans", "dbscan"] = "kmeans"           # "kmeans" or "dbscan"
    n_clusters: int = 5                                         # for KMeans
    eps: float = 0.5                                            # for DBSCAN
    min_samples: int = 10                                       # for DBSCAN
    random_state: int = 42                                      # for reproducibility


class ClusterModelWrapper:
    """Unified clustering model wrapper supporting K-Means and DBSCAN."""

    def __init__(self, config: ClusterConfig):
        self.config = config
        self.model = self._init_model()

    def _init_model(self):
        if self.config.algorithm == "kmeans":
            return KMeans(
                n_clusters=self.config.n_clusters,
                random_state=self.config.random_state
            )
        elif self.config.algorithm == "dbscan":
            return DBSCAN(
                eps=self.config.eps,
                min_samples=self.config.min_samples
            )
        else:
            raise ValueError("Unsupported algorithm. Use 'kmeans' or 'dbscan'.")

    def fit(self, X: pd.DataFrame):
        self.model.fit(X)
        return self

    def predict(self, X: pd.DataFrame):
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        else:
            # DBSCAN doesn't have predict → reassign cluster using fit_transform
            return self.model.fit_predict(X)

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)


class Clusterer:
    def __init__(self, cluster_model: ClusterModelWrapper, df: pd.DataFrame):
        self.cluster_model = cluster_model
        self.df = df


    def fit_and_predict(self) -> np.ndarray:
        X = self.df.drop(columns=["customer_id"]) 
        self.cluster_model.fit(X)
        cluster_labels = self.cluster_model.predict(X)
        df_clusters = self.df[["customer_id"]].copy()
        df_clusters["cluster"] = cluster_labels
        return df_clusters

    def display_metrics(self, df_clusters: pd.DataFrame):
        X = self.df.drop(columns=["customer_id"])
        labels = df_clusters["cluster"].values

        # Some metrics require at least 2 clusters
        if len(set(labels)) > 1 and -1 not in set(labels):  # DBSCAN might have -1 for noise
            sil_score = silhouette_score(X, labels)
            ch_score = calinski_harabasz_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            
            print(f"Silhouette Score: {sil_score:.3f}")
            print(f"Calinski-Harabasz Index: {ch_score:.3f}")
            print(f"Davies-Bouldin Index: {db_score:.3f}")
        else:
            print("Not enough clusters to compute metrics or only noise points (-1) present.")

if __name__ == "__main__":
    df = pd.read_csv("src/synthetic_dataset/generate/llm_generated_customers.csv")
    df_preprocessed = pd.read_csv("src/synthetic_dataset/generate/preprocessed.csv")
    

    # segments = {
    # "female": df[df["gender"]=='female']["customer_id"].tolist(),
    # "male": df[df["gender"]=='male']["customer_id"].tolist()
    # }

    # data_profiler = DataProfiler(client_groups=segments, original_df=df)
    # profile_df = data_profiler.profile_and_plot()
    # print(profile_df)

    #Clustering

    config = ClusterConfig(
    algorithm="dbscan",
    n_clusters=4,
    random_state=42
    )
    cluster_model = ClusterModelWrapper( config=config)
    clusterer = Clusterer(cluster_model=cluster_model, df=df_preprocessed)
    df_clusters = clusterer.fit_and_predict()
    clusterer.display_metrics(df_clusters=df_clusters)

    cluster_segments = {
    str(c): df_clusters[df_clusters["cluster"] == c]["customer_id"].tolist()
    for c in df_clusters["cluster"].unique()
}
    profiler = DataProfiler(client_groups=cluster_segments, original_df=df)
    profile_df = profiler.profile_and_plot()
    