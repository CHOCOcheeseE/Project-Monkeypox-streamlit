import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Library untuk clustering dan analisis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Konfigurasi halaman
st.set_page_config(
    page_title="🔬 Analisis Clustering Monkeypox",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    """Memuat data dari file CSV"""
    try:
        df = pd.read_csv("MonkeyPox.csv")
        return df
    except FileNotFoundError:
        st.error("❌ File 'MonkeyPox.csv' tidak ditemukan! Pastikan file sudah diupload.")
        return None

# Fungsi preprocessing data
@st.cache_data
def preprocess_data(df):
    """Preprocessing data untuk clustering"""
    # Menghapus kolom yang tidak diperlukan
    df_fitur = df.drop(columns=["Patient_ID", "MonkeyPox"])
    
    # Menghandle kolom kategori
    kolom_kategori = df_fitur.select_dtypes(include=["object"]).columns
    
    if len(kolom_kategori) > 0:
        # Mengisi nilai yang hilang
        for col in kolom_kategori:
            if df_fitur[col].isnull().any():
                df_fitur[col] = df_fitur[col].fillna(df_fitur[col].mode()[0])
        
        # Encoding kategori
        df_encoded = pd.get_dummies(df_fitur, columns=kolom_kategori)
    else:
        df_encoded = df_fitur.copy()
    
    # Standardisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    
    # PCA untuk reduksi dimensi
    pca = PCA(n_components=0.95)
    X_processed = pca.fit_transform(X_scaled)
    
    return df_encoded, X_scaled, X_processed, scaler, pca

# Fungsi untuk mencari cluster optimal
@st.cache_data
def find_optimal_clusters(X_processed):
    """Mencari jumlah cluster optimal"""
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
    
    return jumlah_k, inertia_values, silhouette_scores, k_terbaik, score_terbaik

# Fungsi clustering final
@st.cache_data
def perform_clustering(X_processed, k_terbaik):
    """Melakukan clustering final"""
    kmeans_final = KMeans(n_clusters=k_terbaik, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_processed)
    return cluster_labels, kmeans_final

# Main App
def main():
    st.markdown('<h1 class="main-header">🔬 ANALISIS CLUSTERING MONKEYPOX</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Kontrol Analisis")
    st.sidebar.info("Aplikasi ini melakukan analisis clustering pada data monkeypox untuk mengidentifikasi pola dan kelompok pasien.")
    
    # Upload file
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload file CSV",
        type=['csv'],
        help="Upload file 'MonkeyPox.csv' untuk analisis"
    )
    
    # Load data
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = load_data()
    
    if df is None:
        st.warning("⚠️ Silakan upload file 'MonkeyPox.csv' untuk memulai analisis.")
        st.stop()
    
    # Menampilkan informasi data
    st.markdown('<h2 class="section-header">📊 Informasi Dataset</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Jumlah Pasien", df.shape[0])
    with col2:
        st.metric("Jumlah Fitur", df.shape[1])
    with col3:
        positif = (df["MonkeyPox"] == "Positive").sum()
        st.metric("Kasus Positif", positif)
    with col4:
        negatif = (df["MonkeyPox"] == "Negative").sum()
        st.metric("Kasus Negatif", negatif)
    
    # Menampilkan distribusi kasus
    fig_pie = px.pie(
        values=df["MonkeyPox"].value_counts().values,
        names=df["MonkeyPox"].value_counts().index,
        title="Distribusi Kasus Monkeypox",
        color_discrete_sequence=['#ff9999', '#66b3ff']
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Preprocessing data
    with st.spinner("🔄 Memproses data..."):
        df_encoded, X_scaled, X_processed, scaler, pca = preprocess_data(df)
    
    st.success(f"✅ Data berhasil diproses! Dimensi direduksi menjadi {X_processed.shape[1]} komponen.")
    
    # Mencari cluster optimal
    st.markdown('<h2 class="section-header">🎯 Pencarian Cluster Optimal</h2>', unsafe_allow_html=True)
    
    with st.spinner("🔍 Mencari jumlah cluster optimal..."):
        jumlah_k, inertia_values, silhouette_scores, k_terbaik, score_terbaik = find_optimal_clusters(X_processed)
    
    # Visualisasi pemilihan cluster
    fig_cluster = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Elbow Method', 'Silhouette Score'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Elbow method
    fig_cluster.add_trace(
        go.Scatter(x=list(jumlah_k), y=inertia_values, mode='lines+markers',
                  name='Inertia', line=dict(color='blue', width=3)),
        row=1, col=1
    )
    
    # Silhouette score
    fig_cluster.add_trace(
        go.Scatter(x=list(jumlah_k), y=silhouette_scores, mode='lines+markers',
                  name='Silhouette Score', line=dict(color='red', width=3)),
        row=1, col=2
    )
    
    # Menandai cluster terbaik
    fig_cluster.add_vline(x=k_terbaik, line_dash="dash", line_color="green", 
                         annotation_text=f"Optimal: {k_terbaik}", row=1, col=2)
    
    fig_cluster.update_layout(height=400, showlegend=False)
    fig_cluster.update_xaxes(title_text="Jumlah Cluster")
    fig_cluster.update_yaxes(title_text="Inertia", row=1, col=1)
    fig_cluster.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    # Menampilkan hasil optimal
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(f"**🏆 Jumlah Cluster Optimal: {k_terbaik}**")
    st.markdown(f"**📈 Silhouette Score: {score_terbaik:.3f}**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Clustering final
    st.markdown('<h2 class="section-header">🎨 Hasil Clustering</h2>', unsafe_allow_html=True)
    
    with st.spinner("🎨 Melakukan clustering final..."):
        cluster_labels, kmeans_final = perform_clustering(X_processed, k_terbaik)
    
    # Menambahkan hasil clustering ke dataframe
    df["Cluster"] = cluster_labels
    
    # Menampilkan distribusi cluster
    col1, col2 = st.columns(2)
    
    with col1:
        distribusi_cluster = pd.Series(cluster_labels).value_counts().sort_index()
        fig_bar = px.bar(
            x=distribusi_cluster.index,
            y=distribusi_cluster.values,
            title="Distribusi Pasien per Cluster",
            labels={'x': 'Cluster', 'y': 'Jumlah Pasien'},
            color=distribusi_cluster.values,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Tabel silang
        tabel_silang = pd.crosstab(df["Cluster"], df["MonkeyPox"])
        fig_heatmap = px.imshow(
            tabel_silang.values,
            labels=dict(x="Status Monkeypox", y="Cluster", color="Jumlah"),
            x=tabel_silang.columns,
            y=tabel_silang.index,
            title="Matriks Cluster vs Status Monkeypox",
            color_continuous_scale='Blues'
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Evaluasi kualitas clustering
    st.markdown('<h2 class="section-header">📊 Evaluasi Kualitas Clustering</h2>', unsafe_allow_html=True)
    
    silhouette_final = silhouette_score(X_scaled, cluster_labels)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Silhouette Score", f"{silhouette_final:.3f}")
    with col2:
        st.metric("Jumlah Cluster", k_terbaik)
    with col3:
        inertia_final = kmeans_final.inertia_
        st.metric("Inertia", f"{inertia_final:.1f}")
    
    # Tabel persentase
    tabel_persen = pd.crosstab(df["Cluster"], df["MonkeyPox"], normalize="index") * 100
    st.subheader("Persentase Status Monkeypox per Cluster")
    st.dataframe(tabel_persen.round(1), use_container_width=True)
    
    # Visualisasi PCA
    st.markdown('<h2 class="section-header">🔍 Visualisasi PCA</h2>', unsafe_allow_html=True)
    
    # PCA 2D untuk visualisasi
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X_scaled)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pca_cluster = px.scatter(
            x=X_pca[:, 0], y=X_pca[:, 1],
            color=cluster_labels,
            title=f"Hasil Clustering ({k_terbaik} Cluster)",
            labels={
                'x': f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})',
                'y': f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})'
            },
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_pca_cluster, use_container_width=True)
    
    with col2:
        colors = ['red' if x == 'Positive' else 'blue' for x in df['MonkeyPox']]
        fig_pca_actual = px.scatter(
            x=X_pca[:, 0], y=X_pca[:, 1],
            color=df['MonkeyPox'],
            title="Status Monkeypox Asli",
            labels={
                'x': f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})',
                'y': f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})'
            },
            color_discrete_map={'Positive': 'red', 'Negative': 'blue'}
        )
        st.plotly_chart(fig_pca_actual, use_container_width=True)
    
    # Analisis fitur penting
    st.markdown('<h2 class="section-header">🔬 Analisis Fitur Penting</h2>', unsafe_allow_html=True)
    
    # Menghitung rata-rata fitur untuk setiap cluster
    fitur_cluster = df_encoded.copy()
    fitur_cluster["Cluster"] = cluster_labels
    
    rata_rata_cluster = fitur_cluster.groupby("Cluster").mean()
    rata_rata_keseluruhan = df_encoded.mean()
    
    # Mencari fitur penting per cluster
    fitur_penting_per_cluster = {}
    
    for cluster_id in range(k_terbaik):
        rata_cluster = rata_rata_cluster.loc[cluster_id]
        perbedaan = np.abs(rata_cluster - rata_rata_keseluruhan)
        fitur_top = perbedaan.nlargest(5)  # 5 fitur teratas
        fitur_penting_per_cluster[cluster_id] = fitur_top
    
    # Menampilkan fitur penting dalam tabs
    tabs = st.tabs([f"Cluster {i}" for i in range(k_terbaik)])
    
    for i, tab in enumerate(tabs):
        with tab:
            if i in fitur_penting_per_cluster:
                fitur_data = fitur_penting_per_cluster[i]
                
                # Grafik batang horizontal
                fig_fitur = px.bar(
                    x=list(fitur_data.values),
                    y=list(fitur_data.index),
                    orientation='h',
                    title=f"Fitur Paling Membedakan Cluster {i}",
                    labels={'x': 'Selisih dari Rata-rata', 'y': 'Fitur'},
                    color=list(fitur_data.values),
                    color_continuous_scale='viridis'
                )
                fig_fitur.update_layout(height=400)
                st.plotly_chart(fig_fitur, use_container_width=True)
                
                # Tabel detail
                st.subheader(f"Detail Fitur Cluster {i}")
                detail_data = []
                for fitur, selisih in fitur_data.items():
                    nilai_cluster = rata_rata_cluster.loc[i, fitur]
                    nilai_keseluruhan = rata_rata_keseluruhan[fitur]
                    detail_data.append({
                        'Fitur': fitur,
                        'Nilai Cluster': f"{nilai_cluster:.3f}",
                        'Nilai Rata-rata': f"{nilai_keseluruhan:.3f}",
                        'Selisih': f"{selisih:.3f}"
                    })
                
                st.dataframe(pd.DataFrame(detail_data), use_container_width=True)
    
    # Download hasil
    st.markdown('<h2 class="section-header">💾 Download Hasil</h2>', unsafe_allow_html=True)
    
    # Menyiapkan data untuk download
    df_hasil = df.copy()
    csv = df_hasil.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Hasil Clustering (CSV)",
        data=csv,
        file_name=f"monkeypox_clustering_results_{k_terbaik}clusters.csv",
        mime="text/csv"
    )
    
    # Summary
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### 📋 Ringkasan Analisis")
    st.markdown(f"- **Jumlah pasien yang dianalisis**: {len(df)} pasien")
    st.markdown(f"- **Jumlah cluster optimal**: {k_terbaik} cluster")
    st.markdown(f"- **Kualitas clustering (Silhouette Score)**: {silhouette_final:.3f}")
    st.markdown(f"- **Variansi yang dijelaskan PCA**: {sum(pca.explained_variance_ratio_):.1%}")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()