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

# CSS untuk styling
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
        # Coba baca file yang sudah diagregasi
        df = pd.read_csv("mpox_aggregated_data.csv")
        return df
    except FileNotFoundError:
        # Jika file tidak ada, buat data dummy untuk demo
        st.warning("File mpox_aggregated_data.csv tidak ditemukan. Menggunakan data dummy untuk demo.")
        
        # Data dummy untuk demo
        np.random.seed(42)
        countries = ["USA", "Brazil", "Spain", "France", "Germany", "UK", "Canada", "Australia", 
                    "Netherlands", "Belgium", "Portugal", "Italy", "Mexico", "Peru", "Colombia",
                    "Argentina", "Chile", "Sweden", "Norway", "Denmark", "Finland", "Austria",
                    "Switzerland", "Poland", "Czech Republic", "Hungary", "Romania", "Greece",
                    "Turkey", "Israel", "South Africa", "Nigeria", "Kenya", "Ghana", "Morocco",
                    "Egypt", "India", "Thailand", "Philippines", "Japan", "South Korea", "Singapore"]
        
        regions = ["AMRO", "EURO", "AFRO", "SEARO", "WPRO", "EMRO"]
        region_mapping = {
            "USA": "AMRO", "Brazil": "AMRO", "Canada": "AMRO", "Mexico": "AMRO", "Peru": "AMRO",
            "Colombia": "AMRO", "Argentina": "AMRO", "Chile": "AMRO",
            "Spain": "EURO", "France": "EURO", "Germany": "EURO", "UK": "EURO", "Netherlands": "EURO",
            "Belgium": "EURO", "Portugal": "EURO", "Italy": "EURO", "Sweden": "EURO", "Norway": "EURO",
            "Denmark": "EURO", "Finland": "EURO", "Austria": "EURO", "Switzerland": "EURO",
            "Poland": "EURO", "Czech Republic": "EURO", "Hungary": "EURO", "Romania": "EURO",
            "Greece": "EURO", "Turkey": "EURO",
            "South Africa": "AFRO", "Nigeria": "AFRO", "Kenya": "AFRO", "Ghana": "AFRO", "Morocco": "AFRO",
            "India": "SEARO", "Thailand": "SEARO",
            "Australia": "WPRO", "Philippines": "WPRO", "Japan": "WPRO", "South Korea": "WPRO", "Singapore": "WPRO",
            "Egypt": "EMRO", "Israel": "EMRO"
        }
        
        data = []
        for country in countries:
            # Generate realistic data with some correlation
            base_cases = np.random.exponential(100)
            total_confirmed = int(base_cases * np.random.uniform(0.5, 3.0))
            total_deaths = int(total_confirmed * np.random.uniform(0.01, 0.05))
            total_suspected = int(total_confirmed * np.random.uniform(0.1, 0.5))
            case_fatality_rate = (total_deaths / total_confirmed * 100) if total_confirmed > 0 else 0
            
            data.append({
                "country": country,
                "who_region": region_mapping.get(country, "AMRO"),
                "total_confirmed_cases": total_confirmed,
                "total_deaths": total_deaths,
                "total_suspected_cases": total_suspected,
                "case_fatality_rate": case_fatality_rate
            })
        
        df = pd.DataFrame(data)
        return df

# Fungsi untuk preprocessing data
@st.cache_data
def preprocess_data(df, pca_variance=0.95):
    """Preprocessing data untuk clustering"""
    # Menghapus kolom yang tidak diperlukan untuk clustering
    df_fitur = df.drop(columns=["country"])
    
    # Identifikasi kolom kategori dan numerik
    kolom_kategori = df_fitur.select_dtypes(include=["object"]).columns
    kolom_numerik = df_fitur.select_dtypes(include=[np.number]).columns
    
    # One-Hot Encoding untuk kolom kategori
    if len(kolom_kategori) > 0:
        df_encoded = pd.get_dummies(df_fitur, columns=kolom_kategori, drop_first=True)
    else:
        df_encoded = df_fitur.copy()
    
    # Standardisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    
    # PCA untuk reduksi dimensi
    pca = PCA(n_components=pca_variance)
    X_processed = pca.fit_transform(X_scaled)
    
    return X_scaled, X_processed, df_encoded, scaler, pca, kolom_numerik, kolom_kategori

# Fungsi untuk mencari jumlah cluster optimal
@st.cache_data
def find_optimal_clusters(X_processed, max_k=10):
    """Mencari jumlah cluster optimal menggunakan Elbow Method dan Silhouette Score"""
    k_range = range(2, max_k + 1)
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_processed)
        
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(X_processed, labels)
        silhouette_scores.append(sil_score)
    
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
    fig = px.histogram(df_with_clusters, x="Cluster", color="who_region",
                      title="Distribusi WHO Region per Cluster",
                      labels={"count": "Jumlah Negara"})
    fig.update_layout(height=400)
    return fig

def create_numeric_features_plot(df_with_clusters, kolom_numerik):
    """Membuat plot rata-rata fitur numerik per cluster"""
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
    fig = px.scatter(df_with_clusters, 
                    x="total_confirmed_cases", 
                    y="total_deaths",
                    color="Cluster",
                    hover_data=["country", "who_region"],
                    title="Total Confirmed Cases vs Total Deaths by Cluster",
                    labels={"total_confirmed_cases": "Total Confirmed Cases",
                           "total_deaths": "Total Deaths"})
    fig.update_layout(height=500)
    return fig

def create_boxplots(df_with_clusters, kolom_numerik):
    """Membuat box plots untuk setiap fitur numerik per cluster"""
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
    st.write(f"**Silhouette Score Terbaik:** {max(silhouette_scores):.3f}")
    st.write(f"**Jumlah Cluster yang Digunakan:** {n_clusters}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Clustering
    st.markdown('<h2 class="sub-header">🎨 Hasil Clustering</h2>', unsafe_allow_html=True)
    
    with st.spinner("Melakukan clustering..."):
        cluster_labels, kmeans_model = perform_clustering(X_processed, n_clusters)
    
    # Tambahkan hasil cluster ke dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    # Evaluasi clustering
    silhouette_final = silhouette_score(X_scaled, cluster_labels)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Silhouette Score Final", f"{silhouette_final:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        st.write("**Distribusi Cluster:**")
        for cluster_id, count in cluster_counts.items():
            percentage = (count / len(df)) * 100
            st.write(f"Cluster {cluster_id}: {count} negara ({percentage:.1f}%)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualisasi PCA 2D
    st.markdown('<h3>🔍 Visualisasi PCA 2D</h3>', unsafe_allow_html=True)
    fig_pca = create_pca_visualization(X_scaled, cluster_labels, n_clusters)
    st.plotly_chart(fig_pca, use_container_width=True)
    
    # Analisis distribusi
    st.markdown('<h2 class="sub-header">📊 Analisis Distribusi per Cluster</h2>', unsafe_allow_html=True)
    
    # WHO Region distribution
    st.markdown('<h3>🌍 Distribusi WHO Region per Cluster</h3>', unsafe_allow_html=True)
    fig_region = create_cluster_distribution_plot(df_with_clusters)
    st.plotly_chart(fig_region, use_container_width=True)
    
    # Numeric features analysis
    st.markdown('<h3>📈 Rata-rata Fitur Numerik per Cluster</h3>', unsafe_allow_html=True)
    fig_numeric = create_numeric_features_plot(df_with_clusters, kolom_numerik)
    st.plotly_chart(fig_numeric, use_container_width=True)
    
    # Scatter plot
    st.markdown('<h3>🎯 Scatter Plot: Kasus Terkonfirmasi vs Kematian</h3>', unsafe_allow_html=True)
    fig_scatter = create_scatter_plot(df_with_clusters)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Visualisasi tambahan
    st.markdown('<h2 class="sub-header">📊 Visualisasi Tambahan</h2>', unsafe_allow_html=True)
    
    # Tabs untuk visualisasi tambahan
    tab1, tab2, tab3 = st.tabs(["📦 Box Plots", "📊 Histograms", "🔥 Heatmap Profil"])
    
    with tab1:
        st.markdown('<h3>📦 Box Plot Fitur Numerik per Cluster</h3>', unsafe_allow_html=True)
        fig_box = create_boxplots(df_with_clusters, kolom_numerik)
        st.plotly_chart(fig_box, use_container_width=True)
    
    with tab2:
        st.markdown('<h3>📊 Distribusi (Histogram) Fitur Numerik per Cluster</h3>', unsafe_allow_html=True)
        fig_hist = create_histograms(df_with_clusters, kolom_numerik)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab3:
        st.markdown('<h3>🔥 Heatmap Profil Karakteristik Cluster</h3>', unsafe_allow_html=True)
        fig_heatmap = create_heatmap(df_encoded, cluster_labels)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Profil cluster detail
    st.markdown('<h2 class="sub-header">🔬 Profil Detail Cluster</h2>', unsafe_allow_html=True)
    
    # Pilih cluster untuk analisis detail
    selected_cluster = st.selectbox(
        "Pilih Cluster untuk Analisis Detail:",
        options=sorted(df_with_clusters['Cluster'].unique()),
        format_func=lambda x: f"Cluster {x}"
    )
    
    # Filter data untuk cluster yang dipilih
    cluster_data = df_with_clusters[df_with_clusters['Cluster'] == selected_cluster]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<h4>📋 Negara dalam Cluster {selected_cluster}</h4>', unsafe_allow_html=True)
        st.dataframe(cluster_data[['country', 'who_region'] + list(kolom_numerik)], use_container_width=True)
    
    with col2:
        st.markdown(f'<h4>📊 Statistik Cluster {selected_cluster}</h4>', unsafe_allow_html=True)
        cluster_stats = cluster_data[kolom_numerik].describe()
        st.dataframe(cluster_stats, use_container_width=True)
    
    # Download section
    st.markdown('<h2 class="sub-header">💾 Download Data</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Download Data dengan Cluster"):
            st.markdown(get_download_link(df_with_clusters, "mpox_data_with_clusters.csv"), unsafe_allow_html=True)
    
    with col2:
        if st.button("📥 Download Profil Cluster"):
            cluster_profiles = df_with_clusters.groupby('Cluster')[kolom_numerik].agg(['mean', 'std', 'count'])
            st.markdown(get_download_link(cluster_profiles.reset_index(), "cluster_profiles.csv"), unsafe_allow_html=True)
    
    with col3:
        if st.button("📥 Download Statistik Evaluasi"):
            eval_data = pd.DataFrame({
                'k': list(k_range),
                'inertia': inertias,
                'silhouette_score': silhouette_scores
            })
            st.markdown(get_download_link(eval_data, "clustering_evaluation.csv"), unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔬 Aplikasi Analisis Clustering Mpox dengan K-Means</p>
        <p>Dibuat dengan ❤️ menggunakan Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()