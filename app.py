import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import io
import base64

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Analisis Clustering Mpox K-Means",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling - Fixed string formatting
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    """Memuat dan memproses data mpox"""
    try:
        df = pd.read_csv("mpoxcasesbycountryasof30June2024.csv")
        
        # Konversi kolom 'date' ke datetime jika ada
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            # Agregasi data: ambil entri terbaru untuk setiap negara
            df_aggregated = df.sort_values(by="date").drop_duplicates(subset=["country"], keep="last")
        else:
            df_aggregated = df.drop_duplicates(subset=["country"], keep="last")
        
        # Pilih kolom yang relevan untuk clustering
        required_columns = ["country", "total_confirmed_cases", "total_deaths"]
        optional_columns = ["who_region"]
        
        # Check which columns exist
        available_columns = ["country"]
        for col in required_columns[1:]:  # Skip 'country' as it's already added
            if col in df_aggregated.columns:
                available_columns.append(col)
        
        for col in optional_columns:
            if col in df_aggregated.columns:
                available_columns.append(col)
        
        df_final = df_aggregated[available_columns]
        
        # Handle missing values
        df_final = df_final.fillna(0)
        
        return df_final
    except FileNotFoundError:
        st.error("File mpoxcasesbycountryasof30June2024.csv tidak ditemukan. Pastikan file berada di direktori yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Fungsi untuk preprocessing data
@st.cache_data
def preprocess_data(df, pca_variance=0.95):
    """Preprocessing data untuk clustering"""
    # Menghapus kolom yang tidak diperlukan untuk clustering
    df_fitur = df.drop(columns=["country"])
    
    # Identifikasi kolom kategori dan numerik
    kolom_kategori = df_fitur.select_dtypes(include=["object"]).columns.tolist()
    kolom_numerik = df_fitur.select_dtypes(include=[np.number]).columns.tolist()
    
    # One-Hot Encoding untuk kolom kategori
    if len(kolom_kategori) > 0:
        df_encoded = pd.get_dummies(df_fitur, columns=kolom_kategori, drop_first=True)
    else:
        df_encoded = df_fitur.copy()
    
    # Pastikan ada data numerik untuk clustering
    if df_encoded.shape[1] == 0:
        st.error("Tidak ada fitur numerik yang tersedia untuk clustering.")
        st.stop()
    
    # Standardisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    
    # PCA untuk reduksi dimensi
    n_components = min(df_encoded.shape[1], df_encoded.shape[0] - 1)
    pca = PCA(n_components=n_components)
    X_pca_full = pca.fit_transform(X_scaled)
    
    # Pilih komponen berdasarkan variance threshold
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_selected = np.argmax(cumsum_variance >= pca_variance) + 1
    X_processed = X_pca_full[:, :n_components_selected]
    
    return X_scaled, X_processed, df_encoded, scaler, pca, kolom_numerik, kolom_kategori

# Fungsi untuk mencari jumlah cluster optimal
@st.cache_data
def find_optimal_clusters(X_processed, max_k=10):
    """Mencari jumlah cluster optimal menggunakan Elbow Method dan Silhouette Score"""
    n_samples = X_processed.shape[0]
    max_k = min(max_k, n_samples - 1)  # Ensure max_k doesn't exceed reasonable limits
    
    k_range = range(2, max_k + 1)
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        if k < n_samples:  # Ensure k is less than number of samples
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_processed)
            
            inertias.append(kmeans.inertia_)
            sil_score = silhouette_score(X_processed, labels)
            silhouette_scores.append(sil_score)
        else:
            break
    
    # Update k_range to match actual computed values
    k_range = list(range(2, 2 + len(silhouette_scores)))
    
    # Mencari k terbaik berdasarkan silhouette score
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    return k_range, inertias, silhouette_scores, optimal_k

# Fungsi untuk melakukan clustering
@st.cache_data
def perform_clustering(X_processed, n_clusters):
    """Melakukan K-Means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_processed)
    return cluster_labels, kmeans

# Fungsi untuk membuat visualisasi
def create_cluster_selection_plot(k_range, inertias, silhouette_scores, optimal_k):
    """Membuat plot untuk pemilihan jumlah cluster"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Elbow Method', 'Silhouette Score'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Elbow Method
    fig.add_trace(
        go.Scatter(x=list(k_range), y=inertias, mode='lines+markers',
                  name='Inertia', line=dict(color='blue', width=3)),
        row=1, col=1
    )
    
    # Silhouette Score
    fig.add_trace(
        go.Scatter(x=list(k_range), y=silhouette_scores, mode='lines+markers',
                  name='Silhouette Score', line=dict(color='red', width=3)),
        row=1, col=2
    )
    
    # Garis vertikal untuk optimal k
    fig.add_vline(x=optimal_k, line_dash="dash", line_color="green", 
                  annotation_text=f"Optimal: {optimal_k}", row=1, col=2)
    
    fig.update_xaxes(title_text="Jumlah Cluster (k)", row=1, col=1)
    fig.update_xaxes(title_text="Jumlah Cluster (k)", row=1, col=2)
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_pca_visualization(X_scaled, cluster_labels, n_clusters):
    """Membuat visualisasi PCA 2D"""
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X_scaled)
    
    # Clustering pada data PCA untuk visualisasi
    kmeans_2d = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels_2d = kmeans_2d.fit_predict(X_pca)
    centers_2d = kmeans_2d.cluster_centers_
    
    fig = go.Figure()
    
    # Scatter plot untuk setiap cluster
    colors = px.colors.qualitative.Set1[:n_clusters]
    for i in range(n_clusters):
        mask = cluster_labels_2d == i
        fig.add_trace(go.Scatter(
            x=X_pca[mask, 0], y=X_pca[mask, 1],
            mode='markers',
            name=f'Cluster {i}',
            marker=dict(color=colors[i], size=8, opacity=0.7)
        ))
    
    # Pusat cluster
    fig.add_trace(go.Scatter(
        x=centers_2d[:, 0], y=centers_2d[:, 1],
        mode='markers',
        name='Pusat Cluster',
        marker=dict(color='red', size=15, symbol='x', line=dict(width=2, color='black'))
    ))
    
    fig.update_layout(
        title=f'Visualisasi Clustering K-Means (PCA 2D) - {n_clusters} Cluster',
        xaxis_title=f'Komponen 1 ({pca_2d.explained_variance_ratio_[0]:.1%})',
        yaxis_title=f'Komponen 2 ({pca_2d.explained_variance_ratio_[1]:.1%})',
        height=500
    )
    
    return fig

def create_cluster_distribution_plot(df_with_clusters):
    """Membuat plot distribusi WHO Region per cluster"""
    if "who_region" in df_with_clusters.columns:
        fig = px.histogram(df_with_clusters, x="Cluster", color="who_region",
                          title="Distribusi WHO Region per Cluster",
                          labels={"count": "Jumlah Negara"})
        fig.update_layout(height=400)
        return fig
    else:
        # Return empty figure if who_region column doesn't exist
        fig = go.Figure()
        fig.add_annotation(text="WHO Region data tidak tersedia", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400, title="Distribusi WHO Region per Cluster")
        return fig

def create_numeric_features_plot(df_with_clusters, kolom_numerik):
    """Membuat plot rata-rata fitur numerik per cluster"""
    if len(kolom_numerik) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Tidak ada fitur numerik tersedia", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400, title="Rata-rata Fitur Numerik per Cluster")
        return fig
    
    cluster_means = df_with_clusters.groupby('Cluster')[kolom_numerik].mean()
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1[:len(cluster_means)]
    for i, (cluster_id, row) in enumerate(cluster_means.iterrows()):
        fig.add_trace(go.Bar(
            x=kolom_numerik,
            y=row.values,
            name=f'Cluster {cluster_id}',
            marker_color=colors[i]
        ))
    
    fig.update_layout(
        title="Rata-rata Fitur Numerik per Cluster",
        xaxis_title="Fitur",
        yaxis_title="Nilai Rata-rata",
        barmode='group',
        height=500
    )
    
    return fig

def create_scatter_plot(df_with_clusters):
    """Membuat scatter plot kasus vs kematian"""
    if "total_confirmed_cases" in df_with_clusters.columns and "total_deaths" in df_with_clusters.columns:
        hover_data = ["country"]
        if "who_region" in df_with_clusters.columns:
            hover_data.append("who_region")
            
        fig = px.scatter(df_with_clusters, 
                        x="total_confirmed_cases", 
                        y="total_deaths",
                        color="Cluster",
                        hover_data=hover_data,
                        title="Total Confirmed Cases vs Total Deaths by Cluster",
                        labels={"total_confirmed_cases": "Total Confirmed Cases",
                               "total_deaths": "Total Deaths"})
        fig.update_layout(height=500)
        return fig
    else:
        fig = go.Figure()
        fig.add_annotation(text="Data kasus dan kematian tidak tersedia", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400, title="Total Confirmed Cases vs Total Deaths by Cluster")
        return fig

def create_boxplots(df_with_clusters, kolom_numerik):
    """Membuat box plots untuk setiap fitur numerik per cluster"""
    if len(kolom_numerik) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Tidak ada fitur numerik tersedia", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400, title="Box Plots Fitur Numerik per Cluster")
        return fig
    
    n_cols = 2
    n_rows = (len(kolom_numerik) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"Distribusi {col} per Cluster" for col in kolom_numerik],
        vertical_spacing=0.08
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, col in enumerate(kolom_numerik):
        row = (i // n_cols) + 1
        col_pos = (i % n_cols) + 1
        
        for cluster in sorted(df_with_clusters['Cluster'].unique()):
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster][col]
            fig.add_trace(
                go.Box(y=cluster_data, name=f'Cluster {cluster}', 
                      marker_color=colors[cluster % len(colors)],
                      showlegend=(i == 0)),  # Hanya tampilkan legend di subplot pertama
                row=row, col=col_pos
            )
        
        fig.update_yaxes(title_text=col, row=row, col=col_pos)
        fig.update_xaxes(title_text="Cluster", row=row, col=col_pos)
    
    fig.update_layout(height=300 * n_rows, title_text="Box Plot Fitur Numerik per Cluster")
    return fig

def create_histograms(df_with_clusters, kolom_numerik):
    """Membuat histogram untuk setiap fitur numerik per cluster"""
    if len(kolom_numerik) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Tidak ada fitur numerik tersedia", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400, title="Distribusi (Histogram) Fitur Numerik per Cluster")
        return fig
    
    n_cols = 2
    n_rows = (len(kolom_numerik) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"Distribusi {col} per Cluster" for col in kolom_numerik],
        vertical_spacing=0.08
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, col in enumerate(kolom_numerik):
        row = (i // n_cols) + 1
        col_pos = (i % n_cols) + 1
        
        for cluster in sorted(df_with_clusters['Cluster'].unique()):
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster][col]
            fig.add_trace(
                go.Histogram(x=cluster_data, name=f'Cluster {cluster}',
                           marker_color=colors[cluster % len(colors)],
                           opacity=0.7,
                           showlegend=(i == 0)),  # Hanya tampilkan legend di subplot pertama
                row=row, col=col_pos
            )
        
        fig.update_xaxes(title_text=col, row=row, col=col_pos)
        fig.update_yaxes(title_text="Frekuensi", row=row, col=col_pos)
    
    fig.update_layout(height=300 * n_rows, title_text="Distribusi (Histogram) Fitur Numerik per Cluster",
                     barmode='overlay')
    return fig

def create_heatmap(df_encoded, cluster_labels):
    """Membuat heatmap profil cluster"""
    df_with_clusters = df_encoded.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    cluster_profiles = df_with_clusters.groupby('Cluster').mean()
    
    fig = go.Figure(data=go.Heatmap(
        z=cluster_profiles.T.values,
        x=[f'Cluster {i}' for i in cluster_profiles.index],
        y=cluster_profiles.columns,
        colorscale='RdBu_r',
        text=np.round(cluster_profiles.T.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Profil Karakteristik Cluster (Heatmap)",
        xaxis_title="Cluster",
        yaxis_title="Fitur",
        height=600
    )
    
    return fig

# Fungsi untuk download data
def get_download_link(df, filename):
    """Membuat link download untuk dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Analisis Clustering Mpox K-Means</h1>', unsafe_allow_html=True)
    
    # Sidebar untuk kontrol
    st.sidebar.header("⚙️ Pengaturan Analisis")
    
    # Load data
    df = load_data()
    
    # Add download button for the dataset
    st.sidebar.markdown("### Download Dataset")
    st.sidebar.markdown(get_download_link(df, "mpoxcasesbycountryasof30June2024.csv"), unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.subheader("📊 Parameter Clustering")
    
    # PCA variance threshold
    pca_variance = st.sidebar.slider(
        "Threshold Variansi PCA",
        min_value=0.80,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help="Persentase variansi yang dipertahankan dalam PCA"
    )
    
    # Max clusters untuk pencarian optimal
    max_clusters = st.sidebar.slider(
        "Maksimum Jumlah Cluster untuk Evaluasi",
        min_value=5,
        max_value=15,
        value=10,
        help="Rentang maksimum cluster yang akan dievaluasi"
    )
    
    # Manual cluster selection
    manual_clusters = st.sidebar.checkbox("Pilih Jumlah Cluster Manual")
    
    if manual_clusters:
        n_clusters = st.sidebar.slider(
            "Jumlah Cluster",
            min_value=2,
            max_value=10,
            value=3
        )
    
    # Preprocessing
    with st.spinner("Memproses data..."):
        X_scaled, X_processed, df_encoded, scaler, pca, kolom_numerik, kolom_kategori = preprocess_data(df, pca_variance)
    
    # Informasi dataset
    st.markdown('<h2 class="sub-header">📋 Informasi Dataset</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Jumlah Negara", df.shape[0])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Jumlah Fitur Asli", df.shape[1] - 1)  # -1 untuk kolom country
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Fitur Setelah Encoding", df_encoded.shape[1])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Komponen PCA", X_processed.shape[1])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tampilkan data
    if st.checkbox("Tampilkan Data Asli"):
        st.dataframe(df, use_container_width=True)
    
    # Pencarian cluster optimal
    st.markdown('<h2 class="sub-header">🎯 Pencarian Jumlah Cluster Optimal</h2>', unsafe_allow_html=True)
    
    with st.spinner("Mencari jumlah cluster optimal..."):
        k_range, inertias, silhouette_scores, optimal_k = find_optimal_clusters(X_processed, max_clusters)
    
    if not manual_clusters:
        n_clusters = optimal_k
    
    # Plot pemilihan cluster
    fig_selection = create_cluster_selection_plot(k_range, inertias, silhouette_scores, optimal_k)
    st.plotly_chart(fig_selection, use_container_width=True)
    
    # Info cluster optimal
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.write(f"**Jumlah Cluster Optimal (berdasarkan Silhouette Score):** {optimal_k}")
    st.write(f"**Silhouette Score Terbaik:** {silhouette_scores[k_range.index(optimal_k)]:.3f}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Clustering
    st.markdown('<h2 class="sub-header">📊 Hasil Clustering</h2>', unsafe_allow_html=True)
    
    with st.spinner(f"Melakukan clustering dengan {n_clusters} cluster..."):
        cluster_labels, kmeans_model = perform_clustering(X_processed, n_clusters)
        df_with_clusters = df.copy()
        df_with_clusters["Cluster"] = cluster_labels
        
        # Tambahkan pusat cluster ke dataframe untuk analisis
        # Inverse transform pusat cluster untuk mendapatkan nilai dalam skala asli
        try:
            cluster_centers_original_scale = scaler.inverse_transform(pca.inverse_transform(kmeans_model.cluster_centers_))
            
            # Buat DataFrame dari pusat cluster
            cluster_centers_df = pd.DataFrame(cluster_centers_original_scale, columns=df_encoded.columns)
            cluster_centers_df["Cluster"] = range(n_clusters)
            
            st.write("**Profil Pusat Cluster (Skala Asli):**")
            st.dataframe(cluster_centers_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Tidak dapat menampilkan pusat cluster: {str(e)}")
    
    # Visualisasi PCA
    st.markdown('<h3 class="sub-header">Visualisasi PCA</h3>', unsafe_allow_html=True)
    fig_pca = create_pca_visualization(X_scaled, cluster_labels, n_clusters)
    st.plotly_chart(fig_pca, use_container_width=True)
    
    # Visualisasi Distribusi WHO Region per Cluster
    st.markdown('<h3 class="sub-header">Distribusi WHO Region per Cluster</h3>', unsafe_allow_html=True)
    fig_dist = create_cluster_distribution_plot(df_with_clusters)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Visualisasi Rata-rata Fitur Numerik per Cluster
    st.markdown('<h3 class="sub-header">Rata-rata Fitur Numerik per Cluster</h3>', unsafe_allow_html=True)
    fig_numeric_features = create_numeric_features_plot(df_with_clusters, kolom_numerik)
    st.plotly_chart(fig_numeric_features, use_container_width=True)
    
    # Scatter Plot Kasus vs Kematian
    st.markdown('<h3 class="sub-header">Scatter Plot: Total Confirmed Cases vs Total Deaths</h3>', unsafe_allow_html=True)
    fig_scatter = create_scatter_plot(df_with_clusters)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Box Plots Fitur Numerik per Cluster
    st.markdown('<h3 class="sub-header">Box Plots Fitur Numerik per Cluster</h3>', unsafe_allow_html=True)
    fig_boxplots = create_boxplots(df_with_clusters, kolom_numerik)
    st.plotly_chart(fig_boxplots, use_container_width=True)
    
    # Histograms Fitur Numerik per Cluster
    st.markdown('<h3 class="sub-header">Histograms Fitur Numerik per Cluster</h3>', unsafe_allow_html=True)
    fig_histograms = create_histograms(df_with_clusters, kolom_numerik)
    st.plotly_chart(fig_histograms, use_container_width=True)
    
    # Heatmap Profil Karakteristik Cluster
    st.markdown('<h3 class="sub-header">Heatmap Profil Karakteristik Cluster</h3>', unsafe_allow_html=True)
    fig_heatmap = create_heatmap(df_encoded, cluster_labels)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Download results
    st.markdown('<h3 class="sub-header">Download Hasil Clustering</h3>', unsafe_allow_html=True)
    st.markdown(get_download_link(df_with_clusters, "hasil_clustering_mpox.csv"), unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()