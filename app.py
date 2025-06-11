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
from sklearn.manifold import TSNE
import io
import base64

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Analisis Clustering MPOX",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat dan memproses data
@st.cache_data
def load_and_process_data():
    """
    Memuat dataset MPOX dan melakukan preprocessing dengan error handling yang lebih baik
    """
    df = None
    loading_method = None
    error_details = []
    
    # Method 1: Try loading from GitHub
    try:
        st.info("🌐 Mencoba memuat data dari GitHub...")
        url = "https://raw.githubusercontent.com/CHOCOcheeseE/dataset-monkeypox/main/mpox%20cases%20by%20country%20as%20of%2030%20June%202024.csv"
        df = pd.read_csv(url, encoding='utf-8')
        
        # Validate that we actually got data
        if df is not None and not df.empty and len(df.columns) > 1:
            loading_method = "GitHub"
            st.success(f"✅ Berhasil memuat dari GitHub: {df.shape[0]} baris, {df.shape[1]} kolom")
        else:
            df = None
            error_details.append("GitHub: Data kosong atau tidak valid")
            
    except Exception as e:
        error_details.append(f"GitHub: {str(e)}")
        st.warning(f"⚠️ Gagal memuat dari GitHub: {str(e)}")
    
    # Method 2: Try loading from local file if GitHub failed
    if df is None:
        local_filenames = [
            "mpox cases by country as of 30 June 2024.csv",
            "mpox_cases_by_country_as_of_30_June_2024.csv",
            "mpox.csv",
            "dataset.csv"
        ]
        
        for filename in local_filenames:
            try:
                st.info(f"📁 Mencoba memuat file lokal: {filename}")
                df = pd.read_csv(filename, encoding='utf-8')
                
                # Validate the loaded data
                if df is not None and not df.empty and len(df.columns) > 1:
                    loading_method = "Local"
                    st.success(f"✅ Berhasil memuat dari file lokal '{filename}': {df.shape[0]} baris, {df.shape[1]} kolom")
                    break
                else:
                    df = None
                    error_details.append(f"File '{filename}': Data kosong atau tidak valid")
                    
            except FileNotFoundError:
                error_details.append(f"File '{filename}': Tidak ditemukan")
                continue
            except Exception as e:
                error_details.append(f"File '{filename}': {str(e)}")
                continue
    
    # Method 3: Try alternative encoding if still failed
    if df is None:
        try:
            st.info("🔄 Mencoba encoding alternatif...")
            # Try the first local filename with different encoding
            df = pd.read_csv("mpox cases by country as of 30 June 2024.csv", encoding='latin-1')
            
            if df is not None and not df.empty and len(df.columns) > 1:
                loading_method = "Local (latin-1)"
                st.success(f"✅ Berhasil dengan encoding latin-1: {df.shape[0]} baris, {df.shape[1]} kolom")
            else:
                df = None
                
        except Exception as e:
            error_details.append(f"Encoding alternatif: {str(e)}")
    
    # If all methods failed, provide detailed error reporting
    if df is None:
        st.error("❌ Semua metode loading gagal!")
        
        with st.expander("🔍 Detail Error (Klik untuk melihat)"):
            st.write("**Daftar percobaan yang dilakukan:**")
            for i, error in enumerate(error_details, 1):
                st.write(f"{i}. {error}")
        
        st.info("""
        **Solusi yang bisa dicoba:**
        
        **Untuk file lokal:**
        1. Pastikan file CSV berada di folder yang sama dengan script Python Anda
        2. Coba rename file menjadi salah satu dari nama berikut:
           - `mpox cases by country as of 30 June 2024.csv`
           - `mpox_cases_by_country_as_of_30_June_2024.csv`
           - `mpox.csv`
           - `dataset.csv`
        3. Pastikan file tidak sedang dibuka di Excel atau aplikasi lain
        4. Periksa bahwa file memiliki data (bukan file kosong)
        
        **Untuk loading dari internet:**
        1. Periksa koneksi internet Anda
        2. Coba restart aplikasi Streamlit
        3. Coba akses URL secara manual di browser
        
        **Upload manual:**
        Anda juga bisa upload file secara manual menggunakan widget di bawah ini.
        """)
        
        # Provide file uploader as fallback
        st.markdown("### 📤 Upload File CSV Secara Manual")
        uploaded_file = st.file_uploader(
            "Pilih file CSV dataset MPOX", 
            type=['csv'],
            help="Upload file CSV yang berisi data kasus MPOX per negara"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if df is not None and not df.empty:
                    loading_method = "Manual Upload"
                    st.success(f"✅ File berhasil diupload: {df.shape[0]} baris, {df.shape[1]} kolom")
                else:
                    st.error("File yang diupload kosong atau tidak valid")
                    return None, None, None, None, None
            except Exception as e:
                st.error(f"Error membaca file yang diupload: {str(e)}")
                return None, None, None, None, None
        else:
            return None, None, None, None, None
    
    # If we successfully loaded data, proceed with preprocessing
    try:
        st.info(f"🔄 Memproses data yang dimuat dari: {loading_method}")
        
        # Display basic info about the dataset
        st.sidebar.success(f"✅ Dataset dimuat dari: {loading_method}")
        st.sidebar.info(f"📊 Dimensi: {df.shape[0]} negara, {df.shape[1]} kolom")
        
        # Show column names for debugging
        with st.expander("📋 Informasi Kolom Dataset (untuk debugging)"):
            col_info = pd.DataFrame({
                'No': range(1, len(df.columns) + 1),
                'Nama Kolom': df.columns,
                'Tipe Data': df.dtypes.astype(str),
                'Contoh Data': [str(df.iloc[0, i]) if len(df) > 0 else 'N/A' for i in range(len(df.columns))]
            })
            st.dataframe(col_info, use_container_width=True)
        
        # Preprocessing data
        df_fitur = df.drop(columns=["country"] if "country" in df.columns else [])
        
        # Identifikasi kolom kategori dan numerik
        kolom_kategori = df_fitur.select_dtypes(include=["object"]).columns
        kolom_numerik = df_fitur.select_dtypes(include=[np.number]).columns
        
        st.info(f"📊 Fitur numerik: {len(kolom_numerik)}, Fitur kategori: {len(kolom_kategori)}")
        
        # One-hot encoding untuk kolom kategori
        if len(kolom_kategori) > 0:
            df_encoded = pd.get_dummies(df_fitur, columns=kolom_kategori, drop_first=True)
            st.info(f"🔄 One-hot encoding diterapkan pada {len(kolom_kategori)} kolom kategori")
        else:
            df_encoded = df_fitur.copy()
        
        # Handle missing values before standardization
        if df_encoded.isnull().sum().sum() > 0:
            st.warning(f"⚠️ Ditemukan {df_encoded.isnull().sum().sum()} nilai kosong, akan diisi dengan median/modus")
            # Fill numerical columns with median
            for col in df_encoded.select_dtypes(include=[np.number]).columns:
                df_encoded[col].fillna(df_encoded[col].median(), inplace=True)
            # Fill categorical columns with mode
            for col in df_encoded.select_dtypes(include=['object']).columns:
                df_encoded[col].fillna(df_encoded[col].mode()[0] if not df_encoded[col].mode().empty else 'Unknown', inplace=True)
        
        # Standardisasi data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_encoded)
        
        st.success("✅ Preprocessing data selesai!")
        
        return df, df_encoded, X_scaled, kolom_numerik, scaler
        
    except Exception as e:
        st.error(f"❌ Error dalam preprocessing data: {str(e)}")
        with st.expander("🔍 Detail Error Preprocessing"):
            st.exception(e)
        return None, None, None, None, None

# Fungsi untuk mencari jumlah cluster optimal
@st.cache_data
def find_optimal_clusters(X_scaled, max_k=10):
    """
    Mencari jumlah cluster optimal menggunakan Elbow Method dan Silhouette Score
    """
    jumlah_k = range(2, max_k + 1)
    inertia_values = []
    silhouette_scores = []
    
    progress_bar = st.progress(0)
    
    for i, k in enumerate(jumlah_k):
        # Update progress bar
        progress_bar.progress((i + 1) / len(jumlah_k))
        
        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Hitung metrik
        inertia = kmeans.inertia_
        sil_score = silhouette_score(X_scaled, labels)
        
        inertia_values.append(inertia)
        silhouette_scores.append(sil_score)
    
    progress_bar.empty()
    
    # Temukan k optimal berdasarkan silhouette score
    k_optimal = jumlah_k[np.argmax(silhouette_scores)]
    
    return jumlah_k, inertia_values, silhouette_scores, k_optimal

# Fungsi untuk melakukan clustering final
@st.cache_data
def perform_final_clustering(X_scaled, k_optimal):
    """
    Melakukan clustering final dengan k optimal
    """
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Hitung silhouette score final
    silhouette_final = silhouette_score(X_scaled, cluster_labels)
    
    return cluster_labels, silhouette_final, kmeans

# Fungsi untuk PCA
@st.cache_data
def perform_pca(X_scaled, n_components=2):
    """
    Melakukan PCA untuk visualisasi
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, pca

# Fungsi untuk membuat visualisasi
def create_cluster_selection_plot(jumlah_k, inertia_values, silhouette_scores, k_optimal):
    """
    Membuat plot untuk pemilihan jumlah cluster optimal
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Elbow Method', 'Silhouette Score'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Elbow Method
    fig.add_trace(
        go.Scatter(x=list(jumlah_k), y=inertia_values, mode='lines+markers',
                   name='Inertia', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Silhouette Score
    fig.add_trace(
        go.Scatter(x=list(jumlah_k), y=silhouette_scores, mode='lines+markers',
                   name='Silhouette Score', line=dict(color='red')),
        row=1, col=2
    )
    
    # Tandai k optimal
    fig.add_vline(x=k_optimal, line_dash="dash", line_color="green", 
                  annotation_text=f"Optimal: {k_optimal}", row=1, col=2)
    
    fig.update_xaxes(title_text="Jumlah Cluster (k)", row=1, col=1)
    fig.update_xaxes(title_text="Jumlah Cluster (k)", row=1, col=2)
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig

def create_pca_visualization(X_pca, cluster_labels, pca):
    """
    Membuat visualisasi PCA 2D
    """
    # Buat DataFrame untuk plotting
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = cluster_labels
    
    # Buat scatter plot
    fig = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster',
                     title=f'Hasil Clustering K-Means (Visualisasi PCA 2D)',
                     labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                             'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'})
    
    # Tambahkan informasi variance explained
    fig.add_annotation(
        text=f"Total Variance Explained: {sum(pca.explained_variance_ratio_):.1%}",
        xref="paper", yref="paper",
        x=0.02, y=0.98, showarrow=False,
        font=dict(size=12, color="black"),
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    return fig

def create_feature_analysis_plots(df, cluster_labels, kolom_numerik):
    """
    Membuat plot analisis fitur per cluster
    """
    df_plot = df.copy()
    df_plot['Cluster'] = cluster_labels
    
    plots = []
    
    # Box plot untuk setiap fitur numerik
    for col in kolom_numerik:
        fig = px.box(df_plot, x='Cluster', y=col, 
                     title=f'Distribusi {col} per Cluster')
        plots.append(fig)
    
    return plots

# Header aplikasi
st.markdown('<h1 class="main-header">🔬 Analisis Clustering MPOX Kasus per Negara</h1>', 
            unsafe_allow_html=True)

# Sidebar untuk navigasi
st.sidebar.header("🎛️ Panel Kontrol")
selected_section = st.sidebar.selectbox("Pilih Bagian Analisis:", 
                                       ["📊 Gambaran Data", 
                                        "🎯 Penentuan Cluster Optimal", 
                                        "🎨 Hasil Clustering", 
                                        "📈 Analisis Fitur", 
                                        "🔍 Profil Cluster"])

# Memuat data
with st.spinner("Memuat dan memproses data..."):
    df, df_encoded, X_scaled, kolom_numerik, scaler = load_and_process_data()

if df is not None:
    # Bagian 1: Gambaran Data
    if selected_section == "📊 Gambaran Data":
        st.markdown('<h2 class="sub-header">📊 Gambaran Umum Dataset</h2>', 
                    unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Jumlah Negara", df.shape[0])
        with col2:
            st.metric("Jumlah Fitur", df.shape[1])
        with col3:
            st.metric("Data Hilang", df.isnull().sum().sum())
        
        # Tampilkan preview data
        st.markdown("### 👀 Preview Dataset")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Informasi kolom
        st.markdown("### 📋 Informasi Kolom")
        col_info = pd.DataFrame({
            'Kolom': df.columns,
            'Tipe Data': df.dtypes,
            'Nilai Null': df.isnull().sum(),
            'Contoh Nilai': [df[col].iloc[0] if not pd.isna(df[col].iloc[0]) else 'N/A' for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)
        
        # Statistik deskriptif untuk fitur numerik
        if len(kolom_numerik) > 0:
            st.markdown("### 📊 Statistik Deskriptif Fitur Numerik")
            st.dataframe(df[kolom_numerik].describe(), use_container_width=True)
        
        # Distribusi WHO Region
        if 'who_region' in df.columns:
            st.markdown("### 🌍 Distribusi WHO Region")
            region_counts = df['who_region'].value_counts()
            fig = px.bar(x=region_counts.index, y=region_counts.values,
                         title="Jumlah Negara per WHO Region")
            fig.update_xaxes(title="WHO Region")
            fig.update_yaxes(title="Jumlah Negara")
            st.plotly_chart(fig, use_container_width=True)
    
    # Bagian 2: Penentuan Cluster Optimal
    elif selected_section == "🎯 Penentuan Cluster Optimal":
        st.markdown('<h2 class="sub-header">🎯 Penentuan Jumlah Cluster Optimal</h2>', 
                    unsafe_allow_html=True)
        
        # Parameter untuk analisis cluster
        max_k = st.sidebar.slider("Maksimal Jumlah Cluster", 5, 15, 10)
        
        if st.button("🔍 Analisis Cluster Optimal"):
            with st.spinner("Mencari jumlah cluster optimal..."):
                jumlah_k, inertia_values, silhouette_scores, k_optimal = find_optimal_clusters(X_scaled, max_k)
            
            # Tampilkan hasil
            st.success(f"🏆 Jumlah cluster optimal: **{k_optimal}** dengan Silhouette Score: **{max(silhouette_scores):.3f}**")
            
            # Plot pemilihan cluster
            fig = create_cluster_selection_plot(jumlah_k, inertia_values, silhouette_scores, k_optimal)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabel hasil
            results_df = pd.DataFrame({
                'Jumlah Cluster': jumlah_k,
                'Inertia': inertia_values,
                'Silhouette Score': silhouette_scores
            })
            st.markdown("### 📊 Tabel Hasil Evaluasi")
            st.dataframe(results_df, use_container_width=True)
            
            # Simpan k_optimal di session state
            st.session_state.k_optimal = k_optimal
    
    # Bagian 3: Hasil Clustering
    elif selected_section == "🎨 Hasil Clustering":
        st.markdown('<h2 class="sub-header">🎨 Hasil Clustering</h2>', 
                    unsafe_allow_html=True)
        
        # Gunakan k_optimal dari session state atau default
        k_optimal = st.session_state.get('k_optimal', 3)
        
        # Opsi untuk mengubah jumlah cluster
        k_selected = st.sidebar.slider("Jumlah Cluster", 2, 10, k_optimal)
        
        if st.button("🎨 Lakukan Clustering"):
            with st.spinner("Melakukan clustering..."):
                cluster_labels, silhouette_final, kmeans = perform_final_clustering(X_scaled, k_selected)
                X_pca, pca = perform_pca(X_scaled, 2)
            
            # Tampilkan metrik kualitas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Jumlah Cluster", k_selected)
            with col2:
                st.metric("Silhouette Score", f"{silhouette_final:.3f}")
            with col3:
                st.metric("Variance Explained", f"{sum(pca.explained_variance_ratio_):.1%}")
            
            # Distribusi cluster
            st.markdown("### 📊 Distribusi Negara per Cluster")
            cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                             title="Jumlah Negara per Cluster")
                fig.update_xaxes(title="Cluster")
                fig.update_yaxes(title="Jumlah Negara")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(values=cluster_counts.values, names=cluster_counts.index,
                             title="Proporsi Negara per Cluster")
                st.plotly_chart(fig, use_container_width=True)
            
            # Visualisasi PCA
            st.markdown("### 🗺️ Visualisasi PCA 2D")
            fig_pca = create_pca_visualization(X_pca, cluster_labels, pca)
            st.plotly_chart(fig_pca, use_container_width=True)
            
            # Tabel negara per cluster
            st.markdown("### 🌍 Daftar Negara per Cluster")
            df_result = df.copy()
            df_result['Cluster'] = cluster_labels
            
            for cluster_id in sorted(df_result['Cluster'].unique()):
                with st.expander(f"Cluster {cluster_id} ({sum(cluster_labels == cluster_id)} negara)"):
                    cluster_countries = df_result[df_result['Cluster'] == cluster_id]['country'].tolist()
                    st.write(", ".join(cluster_countries))
            
            # Simpan hasil di session state
            st.session_state.cluster_labels = cluster_labels
            st.session_state.df_result = df_result
            st.session_state.silhouette_final = silhouette_final
    
    # Bagian 4: Analisis Fitur
    elif selected_section == "📈 Analisis Fitur":
        st.markdown('<h2 class="sub-header">📈 Analisis Fitur per Cluster</h2>', 
                    unsafe_allow_html=True)
        
        # Cek apakah clustering sudah dilakukan
        if 'cluster_labels' in st.session_state:
            cluster_labels = st.session_state.cluster_labels
            df_result = st.session_state.df_result
            
            # Analisis fitur numerik
            if len(kolom_numerik) > 0:
                st.markdown("### 📊 Distribusi Fitur Numerik per Cluster")
                
                # Pilih fitur untuk dianalisis
                selected_features = st.multiselect(
                    "Pilih fitur untuk dianalisis:",
                    list(kolom_numerik),
                    default=list(kolom_numerik)[:3]  # Default 3 fitur pertama
                )
                
                if selected_features:
                    # Box plots
                    for feature in selected_features:
                        fig = px.box(df_result, x='Cluster', y=feature,
                                     title=f'Distribusi {feature} per Cluster')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabel rata-rata per cluster
                    st.markdown("### 📊 Rata-rata Fitur per Cluster")
                    feature_means = df_result.groupby('Cluster')[selected_features].mean()
                    st.dataframe(feature_means, use_container_width=True)
                    
                    # Heatmap
                    st.markdown("### 🌡️ Heatmap Rata-rata Fitur per Cluster")
                    fig = px.imshow(feature_means.T, 
                                    labels=dict(x="Cluster", y="Fitur", color="Nilai"),
                                    title="Heatmap Rata-rata Fitur per Cluster")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Analisis WHO Region
            if 'who_region' in df_result.columns:
                st.markdown("### 🌍 Distribusi WHO Region per Cluster")
                
                # Crosstab
                crosstab = pd.crosstab(df_result['Cluster'], df_result['who_region'])
                
                # Stacked bar chart
                fig = px.bar(crosstab, title="Distribusi WHO Region per Cluster")
                fig.update_xaxes(title="Cluster")
                fig.update_yaxes(title="Jumlah Negara")
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabel crosstab
                st.dataframe(crosstab, use_container_width=True)
        
        else:
            st.warning("⚠️ Silakan lakukan clustering terlebih dahulu di bagian 'Hasil Clustering'")
    
    # Bagian 5: Profil Cluster
    elif selected_section == "🔍 Profil Cluster":
        st.markdown('<h2 class="sub-header">🔍 Profil Karakteristik Cluster</h2>', 
                    unsafe_allow_html=True)
        
        # Cek apakah clustering sudah dilakukan
        if 'cluster_labels' in st.session_state:
            cluster_labels = st.session_state.cluster_labels
            df_result = st.session_state.df_result
            
            # Profil umum
            st.markdown("### 📋 Ringkasan Profil Cluster")
            
            for cluster_id in sorted(df_result['Cluster'].unique()):
                cluster_data = df_result[df_result['Cluster'] == cluster_id]
                
                with st.expander(f"🔎 Cluster {cluster_id} - {len(cluster_data)} negara"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Karakteristik Numerik:**")
                        if len(kolom_numerik) > 0:
                            cluster_stats = cluster_data[kolom_numerik].agg(['mean', 'std', 'min', 'max'])
                            st.dataframe(cluster_stats.round(2))
                    
                    with col2:
                        st.markdown("**Contoh Negara:**")
                        sample_countries = cluster_data['country'].head(5).tolist()
                        for country in sample_countries:
                            st.write(f"• {country}")
                        
                        if 'who_region' in cluster_data.columns:
                            st.markdown("**WHO Region Dominan:**")
                            region_counts = cluster_data['who_region'].value_counts()
                            st.write(f"• {region_counts.index[0]} ({region_counts.iloc[0]} negara)")
            
            # Perbandingan antar cluster
            st.markdown("### ⚖️ Perbandingan Karakteristik Antar Cluster")
            
            if len(kolom_numerik) > 0:
                # Radar chart
                cluster_means = df_result.groupby('Cluster')[kolom_numerik].mean()
                
                fig = go.Figure()
                
                for cluster_id in cluster_means.index:
                    fig.add_trace(go.Scatterpolar(
                        r=cluster_means.loc[cluster_id].values,
                        theta=cluster_means.columns,
                        fill='toself',
                        name=f'Cluster {cluster_id}'
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                        )),
                    showlegend=True,
                    title="Perbandingan Karakteristik Cluster (Radar Chart)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Download hasil clustering
            st.markdown("### 💾 Unduh Hasil Clustering")
            
            # Konversi ke CSV
            csv_data = df_result.to_csv(index=False)
            
            st.download_button(
                label="📄 Unduh Hasil Clustering (CSV)",
                data=csv_data,
                file_name=f"mpox_clustering_results_{len(set(cluster_labels))}_clusters.csv",
                mime="text/csv"
            )
            
            # Ringkasan kualitas clustering
            st.markdown("### 📊 Kualitas Clustering")
            silhouette_final = st.session_state.get('silhouette_final', 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Silhouette Score", f"{silhouette_final:.3f}")
            with col2:
                st.metric("Jumlah Cluster", len(set(cluster_labels)))
            with col3:
                st.metric("Total Negara", len(cluster_labels))
            
            # Interpretasi hasil
            st.markdown("### 🎯 Interpretasi Hasil")
            
            if silhouette_final > 0.7:
                st.success("🎉 Clustering berkualitas sangat baik! Cluster terbentuk dengan pemisahan yang jelas.")
            elif silhouette_final > 0.5:
                st.info("👍 Clustering berkualitas baik. Terdapat struktur cluster yang dapat diidentifikasi.")
            elif silhouette_final > 0.3:
                st.warning("⚠️ Clustering berkualitas sedang. Mungkin perlu penyesuaian parameter.")
            else:
                st.error("❌ Clustering berkualitas rendah. Pertimbangkan untuk menggunakan metode lain.")
        
        else:
            st.warning("⚠️ Silakan lakukan clustering terlebih dahulu di bagian 'Hasil Clustering'")

# Memuat data dengan error handling yang lebih baik
st.markdown("### 📊 Status Loading Dataset")

with st.spinner("Memuat dan memproses data..."):
    result = load_and_process_data()

# Unpack the results
if result[0] is not None:  # If df is not None
    df, df_encoded, X_scaled, kolom_numerik, scaler = result
    
    # Continue with the rest of your application logic
    # (All your existing sections: Gambaran Data, Penentuan Cluster Optimal, etc.)
    
    # Bagian 1: Gambaran Data
    if selected_section == "📊 Gambaran Data":
        st.markdown('<h2 class="sub-header">📊 Gambaran Umum Dataset</h2>', 
                    unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Jumlah Negara", df.shape[0])
        with col2:
            st.metric("Jumlah Fitur", df.shape[1])
        with col3:
            st.metric("Data Hilang", df.isnull().sum().sum())
        
        # ... rest of your existing code for this section
        
    # Continue with all your other sections exactly as they were
    # (I'm not repeating them all here, but they stay the same)

else:
    # This else block will now rarely be reached because the improved function
    # handles errors more gracefully and provides the file upload option
    st.error("❌ Tidak dapat memuat dataset dari semua sumber yang tersedia.")
    st.info("""
    Aplikasi tidak dapat melanjutkan tanpa dataset. Silakan:
    1. Periksa koneksi internet Anda
    2. Pastikan file CSV tersedia di direktori aplikasi
    3. Gunakan fitur upload manual yang tersedia di atas
    4. Restart aplikasi Streamlit dan coba lagi
    """)