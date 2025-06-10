import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# --- Page Config ---
st.set_page_config(
    page_title="Monkeypox Clustering Analysis",
    page_icon="🦠",
    layout="wide"
)

# --- Styles ---
st.markdown(
    '''
    <style>
    .main { background-color: #f5f5f5; padding: 1rem 2rem; border-radius: 10px; }
    .sidebar .sidebar-content { background-color: #ffffff; border-radius: 10px; padding: 1rem; }
    </style>
    ''',
    unsafe_allow_html=True
)

# --- Title ---
st.title("🔬 Monkeypox Clustering Dashboard")
st.markdown("Analyze clustering patterns in Monkeypox patient data using K-Means and PCA.")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("MonkeyPox.csv")
    return df

df = load_data()

# --- Sidebar ---
st.sidebar.header("Pipeline Settings")
show_raw = st.sidebar.checkbox("Show raw data table", False)
n_clusters = st.sidebar.slider("Number of Clusters", 2, 7, 4)
run_btn = st.sidebar.button("Run Clustering")

# --- Raw Data ---
if show_raw:
    st.subheader("Raw Dataset Preview")
    st.dataframe(df)

# Preprocessing
@st.cache_data
def preprocess(df):
    df_f = df.drop(columns=["Patient_ID", "MonkeyPox"])
    # fill and encode
    objs = df_f.select_dtypes(include=[object]).columns
    if len(objs) > 0:
        for col in objs:
            df_f[col] = df_f[col].fillna(df_f[col].mode()[0])
        df_enc = pd.get_dummies(df_f, columns=objs)
    else:
        df_enc = df_f.copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df_enc)
    return Xs, df_enc

X_scaled, df_encoded = preprocess(df)

# Dimensionality Reduction for visuals
@st.cache_data
def apply_pca(X, n=0.95):
    pca = PCA(n_components=n)
    X_p = pca.fit_transform(X)
    return pca, X_p

pca_full, X_pca_full = apply_pca(X_scaled)
st.sidebar.markdown(f"Explained Variance (95% PCA): {sum(pca_full.explained_variance_ratio_):.1%}")

# Clustering Execution
if run_btn:
    # 1. Find best k automatically
    scores = {}
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_pca_full)
        scores[k] = silhouette_score(X_pca_full, lbl)
    best_k = max(scores, key=scores.get)

    st.subheader("🎯 Optimal Number of Clusters")
    st.write(f"Best k by silhouette: **{best_k}** (Score: {scores[best_k]:.3f})")

    # 2. Final Clustering
    km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km_final.fit_predict(X_pca_full)
    df["Cluster"] = labels

    # Distribution Plot
    st.subheader("📊 Cluster Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=labels, hue=df["MonkeyPox"], palette="Set2", ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.set_title("Patient Distribution per Cluster")
    st.pyplot(fig)

    # Elbow & Silhouette
    st.subheader("📈 Elbow & Silhouette Charts")
    inertia_vals, sil_vals = [], []
    ks = list(scores.keys())
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        inertia_vals.append(km.fit(X_pca_full).inertia_)
        sil_vals.append(scores[k])

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ks, inertia_vals, 'bo-', linewidth=2)
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method")
    ax2.plot(ks, sil_vals, 'ro-', linewidth=2)
    ax2.axvline(x=n_clusters, color='green', linestyle='--')
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Scores")
    st.pyplot(fig2)

    # PCA 2D Visualization
    st.subheader("🔍 2D PCA Clustering View")
    pca2, X2d = PCA(n_components=2).fit_transform(X_scaled), None
    # reuse pca2 incorrectly? Let's compute properly
    pca2 = PCA(n_components=2)
    X2d = pca2.fit_transform(X_scaled)
    km2 = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    lbl2 = km2.fit_predict(X2d)

    fig3, ax3 = plt.subplots()
    scatter = ax3.scatter(X2d[:, 0], X2d[:, 1], c=lbl2, cmap='viridis', alpha=0.7)
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_title("Clusters in PCA Space")
    st.pyplot(fig3)

    # Feature Importance Heatmap
    st.subheader("🌡️ Feature Importance per Cluster")
    df_enc = df_encoded.copy()
    df_enc['Cluster'] = labels
    cluster_means = df_enc.groupby('Cluster').mean().T
    top_feats = cluster_means.apply(lambda col: col.nlargest(3).index.tolist(), axis=0)
    selected = sorted({f for col in top_feats for f in top_feats[col]})[:8]
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cluster_means.loc[selected], annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax4)
    ax4.set_title("Standardized Feature Means by Cluster")
    st.pyplot(fig4)

else:
    st.info("Configure parameters in the sidebar and click 'Run Clustering' to start analysis.")

# Footer
st.markdown("---")
st.markdown("&copy; 2025 Monkeypox Clustering Dashboard")