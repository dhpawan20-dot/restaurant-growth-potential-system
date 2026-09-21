"""
Restaurant Growth Potential Modeling & Strategic Classification System
Analysis pipeline: preprocessing -> PCA -> clustering -> GPI scoring -> labeling

Outputs an enriched CSV with ClusterID, ClusterLabel, GPI, PCA coordinates,
and per-restaurant strategic recommendations, plus a JSON of summary stats
used by the Streamlit dashboard and the research paper.
"""

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, PowerTransformer

RNG_SEED = 42
df = pd.read_csv("/mnt/user-data/outputs/restaurant_growth_dataset.csv")

# ---------------------------------------------------------------------------
# 1. Feature engineering — derived ratios used across scoring & clustering
# ---------------------------------------------------------------------------
df["NetMarginPct"] = df["TotalNetProfit"] / df["TotalRevenue"]
df["AggregatorDependence"] = df["UE_share"] + df["DD_share"]  # of delivery orders
df["AggregatorRevenueShare"] = (
    (df["UberEatsRevenue"] + df["DoorDashRevenue"]) / df["TotalRevenue"]
)
df["CostBurden"] = df["COGSRate"] + df["OPEXRate"]
df["DeliveryEfficiency"] = df["SelfDeliveryOrdersCount"] / (df["DeliveryRadiusKM"] + 1)
df["ScaleIndex"] = df["MonthlyOrders"] * df["GrowthFactor"]
df["RevenuePerOrder"] = df["TotalRevenue"] / df["MonthlyOrders"]

# Skew reduction for heavy-tailed cost/revenue variables
skewed_cols = ["TotalRevenue", "MonthlyOrders", "ScaleIndex", "SD_DeliveryTotalCost"]
pt = PowerTransformer(method="yeo-johnson")
df[[c + "_pt" for c in skewed_cols]] = pt.fit_transform(df[skewed_cols])

# ---------------------------------------------------------------------------
# 2. Feature matrix for clustering (broad, multi-dimensional per the brief)
# ---------------------------------------------------------------------------
feature_cols = [
    "ScaleIndex_pt", "GrowthFactor", "AOV", "RevenuePerOrder",
    "InStoreShare", "UE_share", "DD_share", "SD_share",
    "COGSRate", "OPEXRate", "CommissionRate", "CostBurden",
    "DeliveryRadiusKM", "DeliveryCostOrder", "DeliveryEfficiency",
    "AggregatorDependence", "AggregatorRevenueShare", "NetMarginPct",
]

X = df[feature_cols].copy()
X = X.fillna(X.median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------------
# 3. Dimensionality reduction (PCA) for structure discovery & visualization
# ---------------------------------------------------------------------------
pca = PCA(n_components=5, random_state=RNG_SEED)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]
df["PC3"] = X_pca[:, 2]

# Latent factor loadings (top contributing features per PC) for interpretation
loadings = pd.DataFrame(pca.components_.T, index=feature_cols,
                         columns=[f"PC{i+1}" for i in range(5)])
top_loadings = {
    pc: loadings[pc].abs().sort_values(ascending=False).head(4).index.tolist()
    for pc in ["PC1", "PC2", "PC3"]
}

# ---------------------------------------------------------------------------
# 4. Unsupervised clustering — KMeans (primary), Hierarchical & DBSCAN (checks)
# ---------------------------------------------------------------------------
best_k, best_score, best_labels = None, -1, None
for k in range(3, 7):
    km = KMeans(n_clusters=k, random_state=RNG_SEED, n_init=20)
    labels_k = km.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels_k)
    if score > best_score:
        best_k, best_score, best_labels = k, score, labels_k

kmeans_final = KMeans(n_clusters=best_k, random_state=RNG_SEED, n_init=20)
df["ClusterID"] = kmeans_final.fit_predict(X_pca)

# Robustness checks (not used for final labels, reported for methodology)
hier = AgglomerativeClustering(n_clusters=best_k)
hier_labels = hier.fit_predict(X_pca)
hier_silhouette = silhouette_score(X_pca, hier_labels)

dbscan = DBSCAN(eps=1.5, min_samples=8)
dbscan_labels = dbscan.fit_predict(X_pca)
n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

# ---------------------------------------------------------------------------
# 5. Growth Potential Index (GPI) — composite score, 0-100
# ---------------------------------------------------------------------------
def minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

growth_signal = minmax(df["ScaleIndex"]) * 0.5 + minmax(df["GrowthFactor"]) * 0.5
cost_resilience = 1 - minmax(df["CostBurden"])
channel_balance = 1 - minmax(
    (df[["InStoreShare", "UE_share", "DD_share", "SD_share"]]
     .sub(0.25, axis=0).abs().sum(axis=1))
)  # closer to an even 25/25/25/25 spread across channels = higher balance
logistics_scalability = minmax(df["DeliveryEfficiency"]) * 0.6 + (1 - minmax(df["DeliveryRadiusKM"])) * 0.4
margin_quality = minmax(df["NetMarginPct"])

df["GPI"] = (
    growth_signal * 0.30
    + cost_resilience * 0.20
    + channel_balance * 0.15
    + logistics_scalability * 0.15
    + margin_quality * 0.20
) * 100
df["GPI"] = df["GPI"].round(1)

# ---------------------------------------------------------------------------
# 6. Cluster interpretation & labeling (rule-based on cluster-mean profile)
# ---------------------------------------------------------------------------
label_features = ["GPI", "NetMarginPct", "AggregatorDependence", "ScaleIndex",
                   "CostBurden", "SD_share", "GrowthFactor"]
cluster_profile = df.groupby("ClusterID")[label_features].mean()

overall_median = {
    "GPI": df["GPI"].median(),
    "NetMarginPct": df["NetMarginPct"].median(),
    "AggregatorDependence": df["AggregatorDependence"].median(),
    "ScaleIndex": df["ScaleIndex"].median(),
    "CostBurden": df["CostBurden"].median(),
}

# Standardize cluster-mean profiles ACROSS clusters (z-scores) so archetypes
# are assigned by relative ranking rather than collapsing to a global median.
profile_z = (cluster_profile - cluster_profile.mean()) / (cluster_profile.std() + 1e-9)

archetype_patterns = {
    "High-Growth / High-Risk": {
        "ScaleIndex": 2.0, "GrowthFactor": 1.0, "AggregatorDependence": 0.5, "CostBurden": 0.5,
    },
    "Stable Local Performers": {
        "NetMarginPct": 1.5, "ScaleIndex": -1.0, "CostBurden": -1.0, "AggregatorDependence": -0.5,
    },
    "Aggregator-Dependent Low Margin": {
        "AggregatorDependence": 2.0, "NetMarginPct": -1.5, "SD_share": -1.0,
    },
    "Scalable Self-Delivery Leaders": {
        "SD_share": 2.0, "GPI": 1.0, "AggregatorDependence": -1.0,
    },
    "Overextended, Low Return": {
        "CostBurden": 1.5, "GPI": -2.0, "ScaleIndex": -0.5,
    },
}

from scipy.optimize import linear_sum_assignment

archetype_names = list(archetype_patterns.keys())
cluster_ids = list(profile_z.index)

# Cost matrix: negative alignment (dot product) between each cluster's
# z-score profile and each archetype's ideal pattern -> minimize cost.
cost = np.zeros((len(cluster_ids), len(archetype_names)))
for i, cid in enumerate(cluster_ids):
    for j, name in enumerate(archetype_names):
        pattern = archetype_patterns[name]
        score = sum(profile_z.loc[cid, feat] * weight for feat, weight in pattern.items())
        cost[i, j] = -score

# Pad if fewer clusters than archetypes (or vice versa) so the assignment is square
n = max(len(cluster_ids), len(archetype_names))
padded_cost = np.full((n, n), cost.max() if cost.size else 0)
padded_cost[: cost.shape[0], : cost.shape[1]] = cost

row_ind, col_ind = linear_sum_assignment(padded_cost)
cluster_labels_map = {}
for r, c in zip(row_ind, col_ind):
    if r < len(cluster_ids) and c < len(archetype_names):
        cluster_labels_map[cluster_ids[r]] = archetype_names[c]

df["ClusterLabel"] = df["ClusterID"].map(cluster_labels_map)

# ---------------------------------------------------------------------------
# 7. Strategic recommendation per restaurant
# ---------------------------------------------------------------------------
def recommend(row):
    if row["ClusterLabel"] == "Aggregator-Dependent Low Margin":
        return "Rebalance channels: shift volume toward in-store/self-delivery, renegotiate commission exposure"
    if row["ClusterLabel"] == "Scalable Self-Delivery Leaders":
        return "Optimize: invest in expansion capacity, replicate channel mix in new sites"
    if row["ClusterLabel"] == "Stable Local Performers":
        return "Hold / stabilize: protect margin, selectively test growth levers before scaling"
    if row["ClusterLabel"] == "High-Growth / High-Risk":
        return "Optimize: strong momentum but monitor cost burden; reinforce operational capacity ahead of scale"
    if row["ClusterLabel"] == "Overextended, Low Return":
        return "Hold / stabilize: pause expansion, reduce cost burden and delivery radius before reinvesting"
    return "Hold / stabilize"


df["StrategicRecommendation"] = df.apply(recommend, axis=1)

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
out_csv = "/mnt/user-data/outputs/restaurant_growth_dataset_scored.csv"
df.to_csv(out_csv, index=False)

summary = {
    "n_restaurants": int(len(df)),
    "best_k": int(best_k),
    "kmeans_silhouette": round(float(best_score), 4),
    "hierarchical_silhouette": round(float(hier_silhouette), 4),
    "dbscan_clusters_found": int(n_dbscan_clusters),
    "pca_explained_variance": [round(float(v), 4) for v in explained],
    "top_loadings": top_loadings,
    "cluster_sizes": df["ClusterLabel"].value_counts().to_dict(),
    "cluster_profile_means": df.groupby("ClusterLabel")[
        ["GPI", "NetMarginPct", "AggregatorDependence", "ScaleIndex", "CostBurden", "TotalRevenue", "TotalNetProfit"]
    ].mean().round(3).to_dict(orient="index"),
    "overall_median": {k: round(float(v), 4) for k, v in overall_median.items()},
    "gpi_distribution": {
        "mean": round(float(df["GPI"].mean()), 2),
        "median": round(float(df["GPI"].median()), 2),
        "min": round(float(df["GPI"].min()), 2),
        "max": round(float(df["GPI"].max()), 2),
    },
}

with open("/mnt/user-data/outputs/analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
