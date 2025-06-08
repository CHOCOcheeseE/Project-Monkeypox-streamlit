# ===================================================================
# STREAMLIT VERSION - CLUSTERING MONKEYPOX
# ===================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

st.set_page_config(page_title="Analisis Clustering Monkeypox", layout="wide")
st.title("🔬 Analisis Clustering Monkeypox")

# Memuat data
st.header("📊 Langkah 1: Memuat Data")
try:
    df = pd.read_csv('MonkeyPox.csv')
    st.success(f"Data berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
except FileNotFoundError:
    st.error("File 'MonkeyPox.csv' tidak ditemukan. Harap unggah file tersebut.")
    st.stop()

st.subheader("Distribusi Kasus Monkeypox")
st.write(df['MonkeyPox'].value_counts())
st.write("Data Hilang:", df.isnull().sum().sum())

# Pra-pemrosesan
st.header("🛠️ Langkah 2: Menyiapkan Data")
df_fitur = df.drop(columns=["Patient_ID", "MonkeyPox"])
kolom_kategori = df_fitur.select_dtypes(include=['object']).columns
if len(kolom_kategori) > 0:
    df_encoded = pd.get_dummies(df_fitur, columns=kolom_kategori, drop_first=True)
else:
    df_encoded = df_fitur.copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# Mencari jumlah cluster terbaik
st.header("🎯 Langkah 3: Menentukan Jumlah Cluster Terbaik")
jumlah_k = range(2, 8)
silhouette_scores = []
for k in jumlah_k:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil_score)

k_terbaik = jumlah_k[np.argmax(silhouette_scores)]
st.write(f"Jumlah cluster terbaik: {k_terbaik} (Silhouette Score: {max(silhouette_scores):.3f})")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(jumlah_k, silhouette_scores, marker='o')
ax[0].set_title('Silhouette Score')
ax[0].set_xlabel('Jumlah Cluster')
ax[1].bar(range(len(df_encoded.columns)), np.std(X_scaled, axis=0))
ax[1].set_title('Standar Deviasi Fitur')
st.pyplot(fig)

# Clustering Final
st.header("🎨 Langkah 4: Clustering Akhir dan Visualisasi")
kmeans_final = KMeans(n_clusters=k_terbaik, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

# PCA untuk visualisasi
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
fig2, ax2 = plt.subplots()
scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, ax=ax2)
ax2.set_title('Visualisasi Clustering dengan PCA')
ax2.set_xlabel('Komponen 1')
ax2.set_ylabel('Komponen 2')
st.pyplot(fig2)

# Evaluasi cluster
st.header("📊 Langkah 5: Evaluasi Cluster")
silhouette_final = silhouette_score(X_scaled, df['Cluster'])
st.write(f"Silhouette Score akhir: {silhouette_final:.3f}")
st.dataframe(pd.crosstab(df['Cluster'], df['MonkeyPox']))

# Visualisasi tambahan
st.header("📈 Langkah 6: Visualisasi Tambahan")
fig3, ax3 = plt.subplots()
sns.countplot(data=df, x='Cluster', hue='MonkeyPox', ax=ax3)
ax3.set_title('Distribusi Monkeypox per Cluster')
st.pyplot(fig3)

st.success("Analisis selesai. Gunakan hasil clustering untuk insight lebih lanjut!")