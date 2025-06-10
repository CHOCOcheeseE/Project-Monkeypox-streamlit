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
import warnings
warnings.filterwarnings('ignore')

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Analisis Clustering Monkeypox",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Analisis Clustering Monkeypox")
st.markdown("--- ")

# --- Fungsi untuk memuat data --- 
@st.cache_data
def load_data():
    df = pd.read_csv("MonkeyPox.csv")
    return df

# --- Fungsi untuk preprocessing data ---
@st.cache_data
def preprocess_data(df):
    df_fitur = df.drop(columns=["Patient_ID", "MonkeyPox"])
    
    kolom_kategori = df_fitur.select_dtypes(include=["object"]).columns
    
    if len(kolom_kategori) > 0:
        # Mengisi nilai yang hilang di 'Systemic Illness' dengan mode
        df_fitur["Systemic Illness"] = df_fitur["Systemic Illness"].fillna(df_fitur["Systemic Illness"].mode()[0])
        df_encoded = pd.get_dummies(df_fitur, columns=kolom_kategori)
    else:
        df_encoded = df_fitur.copy()
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    
    pca = PCA(n_components=0.95)
    X_processed = pca.fit_transform(X_scaled)
    
    return X_processed, pca, df_encoded.columns

# --- Fungsi untuk mencari jumlah cluster terbaik ---
@st.cache_data
def find_best_k(X_processed):
    jumlah_k = range(2, 8)
    inertia_values = []
    silhouette_scores = []
    
    for k in jumlah_k:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_processed)
        inertia_values.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_processed, labels))
        
    k_terbaik = jumlah_k[np.argmax(silhouette_scores)]
    score_terbaik = max(silhouette_scores)
    
    return k_terbaik, inertia_values, silhouette_scores, jumlah_k

# --- Main Aplikasi Streamlit ---
def main():
    df = load_data()
    
    st.header("📊 Data Overview")
    st.write("Berikut adalah 5 baris pertama dari dataset Monkeypox:")
    st.dataframe(df.head())
    
    st.write(f"Jumlah pasien: {df.shape[0]}")
    st.write(f"Jumlah kolom: {df.shape[1]}")
    
    st.subheader("Distribusi Kasus Monkeypox")
    kasus = df["MonkeyPox"].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=kasus.index, y=kasus.values, ax=ax, palette="viridis")
    ax.set_title("Distribusi Kasus Monkeypox")
    ax.set_xlabel("Status Monkeypox")
    ax.set_ylabel("Jumlah Pasien")
    st.pyplot(fig)
    
    data_hilang = df.isnull().sum().sum()
    if data_hilang == 0:
        st.success("✅ Bagus! Tidak ada data yang hilang.")
    else:
        st.warning(f"⚠️ Ada {data_hilang} data yang hilang, perlu dibersihkan.")
    
    st.header("🛠️ Data Preprocessing & Dimensionality Reduction")
    X_processed, pca, encoded_cols = preprocess_data(df)
    st.write(f"Data direduksi menjadi {X_processed.shape[1]} komponen, menjelaskan {sum(pca.explained_variance_ratio_):.1%} variansi.")
    
    st.header("🎯 Mencari Jumlah Cluster Terbaik")
    k_terbaik, inertia_values, silhouette_scores, jumlah_k = find_best_k(X_processed)
    
    st.subheader("Elbow Method & Silhouette Score")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_elbow, ax_elbow = plt.subplots()
        ax_elbow.plot(jumlah_k, inertia_values, "bo-", linewidth=2, markersize=8)
        ax_elbow.set_xlabel("Jumlah Cluster")
        ax_elbow.set_ylabel("Inertia (Kepadatan dalam Cluster)")
        ax_elbow.set_title("Elbow Method - Mencari Jumlah Cluster Optimal")
        ax_elbow.grid(True, alpha=0.3)
        ax_elbow.set_xticks(list(jumlah_k))
        st.pyplot(fig_elbow)
        
    with col2:
        fig_silhouette, ax_silhouette = plt.subplots()
        ax_silhouette.plot(jumlah_k, silhouette_scores, "ro-", linewidth=2, markersize=8)
        ax_silhouette.axvline(x=k_terbaik, color="green", linestyle="--", alpha=0.7,
                               label=f"Terbaik: {k_terbaik} cluster")
        ax_silhouette.set_xlabel("Jumlah Cluster")
        ax_silhouette.set_ylabel("Silhouette Score")
        ax_silhouette.set_title("Silhouette Score - Kualitas Pemisahan Cluster")
        ax_silhouette.legend()
        ax_silhouette.grid(True, alpha=0.3)
        ax_silhouette.set_xticks(list(jumlah_k))
        st.pyplot(fig_silhouette)
        
    st.success(f"🏆 Jumlah cluster terbaik berdasarkan Silhouette Score adalah: {k_terbaik} dengan score: {max(silhouette_scores):.3f}")
    
    st.header("✨ Hasil Clustering")
    kmeans = KMeans(n_clusters=k_terbaik, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_processed)
    
    st.subheader("Ukuran Masing-masing Cluster")
    cluster_sizes = df["Cluster"].value_counts().sort_index()
    st.dataframe(cluster_sizes.reset_index().rename(columns={'index': 'Cluster', 'count': 'Jumlah Pasien'}))
    
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(cluster_sizes, labels=cluster_sizes.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    ax_pie.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    ax_pie.set_title("Distribusi Pasien per Cluster")
    st.pyplot(fig_pie)
    
    st.subheader("Karakteristik Cluster")
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=encoded_cols)
    st.write("Rata-rata nilai fitur untuk setiap cluster (setelah standardisasi dan PCA):")
    st.dataframe(cluster_centers)
    
    st.markdown("**Analisis Karakteristik Cluster (Berdasarkan Fitur Asli):**")
    # Convert boolean columns to int for mean calculation
    boolean_cols = [col for col in df.columns if df[col].dtype == 'bool']
    for col in boolean_cols:
        df[col] = df[col].astype(int)

    # Group by cluster and calculate mean for relevant features
    cluster_summary = df.groupby('Cluster')[['Systemic Illness', 'Rectal Pain', 'Sore Throat', 'Penile Oedema', 
                                            'Oral Lesions', 'Solitary Lesion', 'Swollen Tonsils', 
                                            'HIV Infection', 'Sexually Transmitted Infection']].mean()
    st.dataframe(cluster_summary)

    st.subheader("Visualisasi Cluster (PCA 2D)")
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_processed)
    df_pca_2d = pd.DataFrame(X_pca_2d, columns=["PC1", "PC2"])
    df_pca_2d["Cluster"] = df["Cluster"]
    
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        x="PC1", y="PC2",
        hue="Cluster",
        palette="viridis",
        data=df_pca_2d,
        legend="full",
        alpha=0.7,
        ax=ax_scatter
    )
    
    # Plot centroids
    centroids_2d = pca_2d.transform(kmeans.cluster_centers_)
    ax_scatter.scatter(
        centroids_2d[:, 0], centroids_2d[:, 1],
        marker="X", s=200, color="red", label="Centroids", zorder=10
    )
    
    ax_scatter.set_title("Visualisasi Cluster dengan PCA (2 Komponen Utama)")
    ax_scatter.set_xlabel("Principal Component 1")
    ax_scatter.set_ylabel("Principal Component 2")
    ax_scatter.legend()
    st.pyplot(fig_scatter)
    
    st.header("🌳 Hierarchical Clustering (Dendrogram)")
    st.write("Dendrogram membantu memvisualisasikan hierarki cluster. Perhatikan bahwa untuk dataset besar, dendrogram mungkin terlihat sangat padat.")
    
    # Limit the number of samples for dendrogram for better visualization with large datasets
    # Taking a random sample of 1000 points if dataset is too large
    if X_processed.shape[0] > 1000:
        np.random.seed(42)
        sample_indices = np.random.choice(X_processed.shape[0], 1000, replace=False)
        X_dendrogram = X_processed[sample_indices]
    else:
        X_dendrogram = X_processed

    linked = linkage(X_dendrogram, method='ward')
    
    fig_dendro = plt.figure(figsize=(15, 7))
    dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True)
    plt.title('Dendrogram Hierarchical Clustering')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    st.pyplot(fig_dendro)
    
    st.markdown("--- ")
    st.info("Aplikasi ini dibuat untuk tugas akhir analisis clustering Monkeypox.")

if __name__ == "__main__":
    main()