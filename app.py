"""
Restaurant Growth Potential Modeling & Strategic Classification System
SkyCity Auckland Restaurants & Bars — Streamlit Dashboard

Run with: streamlit run app.py
Expects restaurant_growth_dataset_scored.csv in the same folder
(produced by analysis_pipeline.py).
"""

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Restaurant Growth Potential System",
    page_icon="📈",
    layout="wide",
)

CLUSTER_COLORS = {
    "High-Growth / High-Risk": "#E4572E",
    "Stable Local Performers": "#3B8686",
    "Aggregator-Dependent Low Margin": "#A44A3F",
    "Scalable Self-Delivery Leaders": "#2E86AB",
    "Overextended, Low Return": "#7A7A7A",
}


@st.cache_data
def load_data():
    df = pd.read_csv("restaurant_growth_dataset_scored.csv")
    return df


@st.cache_data
def load_summary():
    with open("analysis_summary.json") as f:
        return json.load(f)


df = load_data()
summary = load_summary()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.title("📊 Filters")
subregions = st.sidebar.multiselect("Subregion", sorted(df["Subregion"].unique()))
cuisines = st.sidebar.multiselect("Cuisine Type", sorted(df["CuisineType"].unique()))
segments_sel = st.sidebar.multiselect("Segment", sorted(df["Segment"].unique()))
clusters_sel = st.sidebar.multiselect("Cluster / Archetype", sorted(df["ClusterLabel"].unique()))

filtered = df.copy()
if subregions:
    filtered = filtered[filtered["Subregion"].isin(subregions)]
if cuisines:
    filtered = filtered[filtered["CuisineType"].isin(cuisines)]
if segments_sel:
    filtered = filtered[filtered["Segment"].isin(segments_sel)]
if clusters_sel:
    filtered = filtered[filtered["ClusterLabel"].isin(clusters_sel)]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(filtered)}** of {len(df)} restaurant branches")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📈 Restaurant Growth Potential & Strategic Classification")
st.caption("SkyCity Auckland Restaurants & Bars — data-driven growth readiness and strategic segmentation")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Branches Analyzed", len(filtered))
k2.metric("Avg Growth Potential Index", f"{filtered['GPI'].mean():.1f}")
k3.metric("Avg Net Margin", f"{filtered['NetMarginPct'].mean()*100:.1f}%")
k4.metric("Avg Aggregator Dependence", f"{filtered['AggregatorDependence'].mean()*100:.1f}%")

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Cluster Map", "🏆 Growth Scorecards", "📡 Feature Radar", "🧭 Strategy Recommendations"
])

# --- Tab 1: Cluster map -----------------------------------------------------
with tab1:
    st.subheader("Restaurant Cluster Map (PCA projection)")
    st.caption(
        f"KMeans (k={summary['best_k']}) on PCA-reduced features · "
        f"silhouette={summary['kmeans_silhouette']} · "
        f"PC1+PC2 explain {sum(summary['pca_explained_variance'][:2])*100:.1f}% of variance"
    )
    fig = px.scatter(
        filtered, x="PC1", y="PC2",
        color="ClusterLabel",
        color_discrete_map=CLUSTER_COLORS,
        size="ScaleIndex",
        hover_data=["RestaurantName", "Subregion", "Segment", "GPI", "NetMarginPct"],
        labels={"PC1": "Latent Factor 1 (scale / cost)", "PC2": "Latent Factor 2 (channel dependence / margin)"},
        height=550,
    )
    fig.update_layout(legend_title_text="Archetype")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Geographic distribution by subregion")
    geo = filtered.groupby(["Subregion", "ClusterLabel"]).size().reset_index(name="Count")
    fig_geo = px.bar(
        geo, x="Subregion", y="Count", color="ClusterLabel",
        color_discrete_map=CLUSTER_COLORS, height=400,
    )
    st.plotly_chart(fig_geo, use_container_width=True)

# --- Tab 2: Growth scorecards ------------------------------------------------
with tab2:
    st.subheader("Growth Potential Scorecards")
    sort_col = st.selectbox("Sort by", ["GPI", "TotalRevenue", "TotalNetProfit", "NetMarginPct", "ScaleIndex"], index=0)
    top_n = st.slider("Number of restaurants to show", 5, 50, 15)

    top_df = filtered.sort_values(sort_col, ascending=False).head(top_n)
    for _, row in top_df.iterrows():
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                st.markdown(f"**{row['RestaurantName']}**")
                st.caption(f"{row['Segment']} · {row['CuisineType']} · {row['Subregion']}")
                st.markdown(
                    f"<span style='background-color:{CLUSTER_COLORS.get(row['ClusterLabel'],'#888')};"
                    f"color:white;padding:2px 8px;border-radius:10px;font-size:0.8em'>{row['ClusterLabel']}</span>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.metric("GPI", f"{row['GPI']:.1f} / 100")
                st.metric("Net Margin", f"{row['NetMarginPct']*100:.1f}%")
            with c3:
                st.metric("Monthly Orders", f"{row['MonthlyOrders']:,}")
                st.metric("Total Net Profit", f"${row['TotalNetProfit']:,.0f}")

    st.markdown("##### GPI distribution")
    fig_hist = px.histogram(filtered, x="GPI", color="ClusterLabel", nbins=25,
                             color_discrete_map=CLUSTER_COLORS, height=350)
    st.plotly_chart(fig_hist, use_container_width=True)

# --- Tab 3: Feature contribution radar --------------------------------------
with tab3:
    st.subheader("Feature Contribution Radar — Compare Clusters")
    radar_features = ["GPI", "NetMarginPct", "AggregatorDependence", "CostBurden", "SD_share", "GrowthFactor"]
    radar_df = df.groupby("ClusterLabel")[radar_features].mean()

    # Normalize each feature 0-1 across clusters for comparable radar axes
    radar_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-9)

    fig_radar = go.Figure()
    clusters_to_plot = clusters_sel if clusters_sel else radar_df.index.tolist()
    for cluster in clusters_to_plot:
        if cluster in radar_norm.index:
            values = radar_norm.loc[cluster].tolist()
            values += values[:1]
            fig_radar.add_trace(go.Scatterpolar(
                r=values, theta=radar_features + [radar_features[0]],
                fill="toself", name=cluster,
                line_color=CLUSTER_COLORS.get(cluster, "#888"),
            ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=550, showlegend=True,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("##### Compare individual restaurants")
    sel_names = st.multiselect(
        "Pick up to 4 restaurants", filtered["RestaurantName"].tolist(),
        default=filtered["RestaurantName"].tolist()[:2] if len(filtered) >= 2 else filtered["RestaurantName"].tolist(),
        max_selections=4,
    )
    if sel_names:
        indiv = filtered[filtered["RestaurantName"].isin(sel_names)]
        indiv_norm = indiv.copy()
        for feat in radar_features:
            lo, hi = df[feat].min(), df[feat].max()
            indiv_norm[feat] = (indiv[feat] - lo) / (hi - lo + 1e-9)
        fig_radar2 = go.Figure()
        for _, r in indiv_norm.iterrows():
            vals = r[radar_features].tolist()
            vals += vals[:1]
            fig_radar2.add_trace(go.Scatterpolar(
                r=vals, theta=radar_features + [radar_features[0]],
                fill="toself", name=r["RestaurantName"],
            ))
        fig_radar2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=500)
        st.plotly_chart(fig_radar2, use_container_width=True)

# --- Tab 4: Strategy recommendations ----------------------------------------
with tab4:
    st.subheader("Strategic Recommendation Panel")

    rec_counts = filtered["StrategicRecommendation"].str.split(":").str[0].value_counts()
    fig_rec = px.pie(
        names=rec_counts.index, values=rec_counts.values, height=350,
        title="Recommendation mix across filtered branches",
    )
    st.plotly_chart(fig_rec, use_container_width=True)

    st.markdown("##### Cluster-level playbook")
    playbook = {
        "High-Growth / High-Risk": "Strong scale + growth momentum, but rising cost burden. **Optimize**: reinforce ops capacity, protect margin ahead of further scale.",
        "Stable Local Performers": "Healthy margins, moderate scale. **Hold / stabilize**: defend current position, pilot growth levers selectively.",
        "Aggregator-Dependent Low Margin": "Heavy 3rd-party delivery reliance eroding margin. **Rebalance channels**: shift volume to in-store/self-delivery, renegotiate commissions.",
        "Scalable Self-Delivery Leaders": "Efficient self-delivery, healthy margin. **Optimize**: strong candidates for replication in new sites.",
        "Overextended, Low Return": "High cost burden, weak growth potential. **Hold / stabilize**: pause expansion, cut cost burden and delivery radius first.",
    }
    for cl, text in playbook.items():
        if cl in filtered["ClusterLabel"].unique():
            with st.expander(f"{cl}  ({(filtered['ClusterLabel']==cl).sum()} branches)"):
                st.markdown(text)

    st.markdown("##### Restaurant-level detail")
    detail_cols = [
        "RestaurantName", "Subregion", "Segment", "ClusterLabel", "GPI",
        "NetMarginPct", "AggregatorDependence", "StrategicRecommendation",
    ]
    st.dataframe(
        filtered[detail_cols].sort_values("GPI", ascending=False).reset_index(drop=True),
        use_container_width=True, height=400,
    )

st.markdown("---")
st.caption(
    "Growth Potential Index (GPI) = 30% growth signal + 20% cost resilience + 15% channel balance "
    "+ 15% logistics scalability + 20% margin quality, each min-max normalized 0-1 across all branches."
)
