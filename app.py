import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Konfigurasi Streamlit
st.set_page_config(page_title="Clustering Monkeypox", layout="wide")
st.title("🧬 Analisis Clustering pada Data Monkeypox")

# Upload file CSV
uploaded_file = st.file_uploader("📂 Upload dataset MonkeyPox.csv", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data berhasil dimuat!")

    # Tampilkan ringkasan data
    st.subheader("👀 Sekilas tentang data")
    st.dataframe(df.head())

    # Preprocessing
    st.subheader("🔍 Preprocessing")
    X = df.select_dtypes(include=[np.number])
    st.write(f"📐 Menggunakan {X.shape[1]} fitur numerik untuk clustering.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    st.subheader("📊 Reduksi Dimensi dengan PCA")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    st.write(f"✨ Variansi yang dijelaskan oleh 2 komponen: {np.sum(pca.explained_variance_ratio_):.2%}")
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

    # Pilih jumlah klaster
    k = st.slider("🔢 Pilih jumlah klaster (KMeans)", min_value=2, max_value=10, value=3)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)
    df_pca["Cluster"] = cluster_labels

    # Visualisasi hasil clustering
    st.subheader("🌀 Visualisasi Clustering")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Cluster", palette="husl", s=70, ax=ax)
    ax.set_title(f"KMeans Clustering (k={k})")
    st.pyplot(fig)

    # Silhouette Score
    score = silhouette_score(X_pca, cluster_labels)
    st.info(f"📈 Silhouette Score untuk k={k}: **{score:.4f}**")

    # Elbow Method
    st.subheader("📉 Metode Elbow")
    distortions = []
    K = range(2, 11)
    for i in K:
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(X_pca)
        distortions.append(km.inertia_)
    fig2, ax2 = plt.subplots()
    ax2.plot(K, distortions, 'bx-')
    ax2.set_xlabel('Jumlah Klaster')
    ax2.set_ylabel('Inertia')
    ax2.set_title('Metode Elbow')
    st.pyplot(fig2)

    # Dendrogram
    st.subheader("🌳 Dendrogram (Hierarchical Clustering)")
    linked = linkage(X_scaled, 'ward')
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    dendrogram(linked, truncate_mode='level', p=5, ax=ax3)
    st.pyplot(fig3)

else:
    st.warning("👈 Silakan upload file `MonkeyPox.csv` terlebih dahulu.")
