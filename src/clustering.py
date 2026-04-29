import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def run_clustering(data_path="data/clustering_data.csv", n_clusters=3):
    df = pd.read_csv(data_path)

    X = df.select_dtypes(include="number")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(X_scaled)

    df["cluster"] = clusters

    plt.figure(figsize=(8, 6))
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters)
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.title("Customer Segmentation via K-Means Clustering")
    os.makedirs("outputs/visuals", exist_ok=True)
    plt.savefig("outputs/visuals/clustering_results.png", dpi=300, bbox_inches="tight")
    plt.close()

    df.to_csv("outputs/clustering_output.csv", index=False)

    return df