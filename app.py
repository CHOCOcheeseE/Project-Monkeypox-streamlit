import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Library untuk clustering dan analisis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Clustering Monkeypox",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem 0;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}

.metric-container {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}

.sidebar .sidebar-content {
    background: #f8f9fa;
}
</style>
""", unsafe_allow_html=True)

# Header utama
st.markdown("""
<div class="main-header">
    <h1>🔬 ANALISIS CLUSTERING MONKEYPOX</h1>
    <p>Aplikasi Analisis Clustering untuk Identifikasi Pola Kasus Monkeypox</p>
</div>
""", unsafe_allow_html=True)

# Sidebar untuk kontrol
st.sidebar.title("⚙️ Pengaturan Analisis")
st.sidebar.markdown("---")

# Upload file
uploaded_file = st.sidebar.file_uploader(
    "📁 Upload File Dataset (CSV)", 
    type=['csv'],
    help="Upload file MonkeyPox.csv"
)

# Fungsi untuk memuat data
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Fungsi untuk preprocessing
@st.cache_data
def preprocess_data(df):
    # Menghapus kolom yang tidak diperlukan
    df_fitur = df.drop(columns=["Patient_ID", "MonkeyPox"])
    
    # Encoding data kategori
    kolom_kategori = df_fitur.select_dtypes(include=['object']).columns
    if len(kolom_kategori) > 0:
        df_encoded = pd.get_dummies(df_fitur, columns=kolom_kategori, drop_first=True)
    else:
        df_encoded = df_fitur.copy()
    
    # Standardisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    
    return df_encoded, X_scaled, scaler

# Fungsi untuk mencari cluster optimal
@st.cache_data
def find_optimal_clusters(X_scaled, max_k=8):
    jumlah_k = range(2, max_k)
    inertia_values = []
    silhouette_scores = []
    
    for k in jumlah_k:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        inertia = kmeans.inertia_
        sil_score = silhouette_score(X_scaled, labels)
        
        inertia_values.append(inertia)
        silhouette_scores.append(sil_score)
    
    k_terbaik = jumlah_k[np.argmax(silhouette_scores)]
    return jumlah_k, inertia_values, silhouette_scores, k_terbaik

# Fungsi untuk melakukan clustering
@st.cache_data
def perform_clustering(X_scaled, k_terbaik):
    kmeans = KMeans(n_clusters=k_terbaik, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    return kmeans, cluster_labels

# Main application
if uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file)
    
    # Sidebar info
    st.sidebar.success(f"✅ Data berhasil dimuat!")
    st.sidebar.info(f"📊 Jumlah pasien: {df.shape[0]}")
    st.sidebar.info(f"📋 Jumlah kolom: {df.shape[1]}")
    
    # Parameter clustering
    st.sidebar.markdown("### 🎯 Parameter Clustering")
    max_clusters = st.sidebar.slider("Maksimal Jumlah Cluster", 3, 10, 8)
    show_detailed = st.sidebar.checkbox("Tampilkan Analisis Detail", value=True)
    show_comparison = st.sidebar.checkbox("Bandingkan dengan Hierarchical", value=False)
    
    # Tabs untuk organisasi konten
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview Data", 
        "🎯 Pemilihan Cluster", 
        "🔍 Hasil Clustering", 
        "📈 Visualisasi", 
        "💡 Interpretasi"
    ])
    
    with tab1:
        st.header("📊 Overview Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Informasi Dataset")
            st.write(f"**Jumlah Pasien:** {df.shape[0]}")
            st.write(f"**Jumlah Fitur:** {df.shape[1]}")
            
            # Distribusi kasus
            st.subheader("Distribusi Kasus Monkeypox")
            kasus_dist = df['MonkeyPox'].value_counts()
            
            fig_pie = px.pie(
                values=kasus_dist.values, 
                names=kasus_dist.index,
                title="Distribusi Status Monkeypox",
                color_discrete_sequence=['#FF6B6B', '#4ECDC4']
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("Preview Data")
            st.dataframe(df.head(), use_container_width=True)
            
            # Info data hilang
            data_hilang = df.isnull().sum().sum()
            if data_hilang == 0:
                st.success("✅ Tidak ada data yang hilang")
            else:
                st.warning(f"⚠️ Ada {data_hilang} data yang hilang")
    
    with tab2:
        st.header("🎯 Pemilihan Jumlah Cluster Optimal")
        
        # Preprocessing
        with st.spinner("Memproses data..."):
            df_encoded, X_scaled, scaler = preprocess_data(df)
            jumlah_k, inertia_values, silhouette_scores, k_terbaik = find_optimal_clusters(X_scaled, max_clusters)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="🏆 Jumlah Cluster Terbaik", 
                value=k_terbaik,
                help="Berdasarkan Silhouette Score tertinggi"
            )
            
        with col2:
            st.metric(
                label="📊 Silhouette Score", 
                value=f"{max(silhouette_scores):.3f}",
                help="Mengukur kualitas pemisahan cluster"
            )
        
        # Grafik pemilihan cluster
        fig_selection = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Elbow Method', 'Silhouette Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Elbow method
        fig_selection.add_trace(
            go.Scatter(x=list(jumlah_k), y=inertia_values, mode='lines+markers',
                      name='Inertia', line=dict(color='blue', width=3)),
            row=1, col=1
        )
        
        # Silhouette score
        fig_selection.add_trace(
            go.Scatter(x=list(jumlah_k), y=silhouette_scores, mode='lines+markers',
                      name='Silhouette Score', line=dict(color='red', width=3)),
            row=1, col=2
        )
        
        # Highlight cluster terbaik
        fig_selection.add_vline(x=k_terbaik, line_dash="dash", line_color="green", 
                               annotation_text=f"Optimal: {k_terbaik}", row=1, col=2)
        
        fig_selection.update_xaxes(title_text="Jumlah Cluster", row=1, col=1)
        fig_selection.update_xaxes(title_text="Jumlah Cluster", row=1, col=2)
        fig_selection.update_yaxes(title_text="Inertia", row=1, col=1)
        fig_selection.update_yaxes(title_text="Silhouette Score", row=1, col=2)
        
        fig_selection.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_selection, use_container_width=True)
    
    with tab3:
        st.header("🔍 Hasil Clustering")
        
        # Perform clustering
        with st.spinner("Melakukan clustering..."):
            kmeans_final, cluster_labels = perform_clustering(X_scaled, k_terbaik)
            df_result = df.copy()
            df_result['Cluster'] = cluster_labels
        
        # Distribusi cluster
        st.subheader("Distribusi Pasien per Cluster")
        distribusi_cluster = pd.Series(cluster_labels).value_counts().sort_index()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            for cluster_id, jumlah in distribusi_cluster.items():
                persentase = (jumlah / len(df)) * 100
                st.write(f"**Cluster {cluster_id}:** {jumlah} pasien ({persentase:.1f}%)")
        
        with col2:
            fig_dist = px.bar(
                x=distribusi_cluster.index, 
                y=distribusi_cluster.values,
                title="Distribusi Pasien per Cluster",
                labels={'x': 'Cluster', 'y': 'Jumlah Pasien'},
                color=distribusi_cluster.values,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Tabel silang
        st.subheader("Analisis Cluster vs Status Monkeypox")
        tabel_silang = pd.crosstab(df_result['Cluster'], df_result['MonkeyPox'], margins=True)
        st.dataframe(tabel_silang, use_container_width=True)
        
        # Persentase
        tabel_persen = pd.crosstab(df_result['Cluster'], df_result['MonkeyPox'], normalize='index') * 100
        st.subheader("Persentase Status Monkeypox per Cluster")
        st.dataframe(tabel_persen.round(1), use_container_width=True)
    
    with tab4:
        st.header("📈 Visualisasi Clustering")
        
        # PCA untuk visualisasi
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        st.info(f"PCA menjelaskan {sum(pca.explained_variance_ratio_):.1%} dari total variasi data")
        
        # Visualisasi PCA
        col1, col2 = st.columns(2)
        
        with col1:
            # Clustering results
            fig_cluster = px.scatter(
                x=X_pca[:, 0], y=X_pca[:, 1], 
                color=cluster_labels,
                title="Hasil Clustering K-Means",
                labels={'x': f'Komponen 1 ({pca.explained_variance_ratio_[0]:.1%})', 
                       'y': f'Komponen 2 ({pca.explained_variance_ratio_[1]:.1%})'},
                color_continuous_scale='viridis'
            )
            
            # Tambahkan centroid
            centers_pca = pca.transform(kmeans_final.cluster_centers_)
            fig_cluster.add_scatter(
                x=centers_pca[:, 0], y=centers_pca[:, 1],
                mode='markers', marker=dict(symbol='x', size=15, color='red'),
                name='Centroid'
            )
            
            st.plotly_chart(fig_cluster, use_container_width=True)
        
        with col2:
            # Status asli
            colors_status = ['red' if x == 'Positive' else 'blue' for x in df['MonkeyPox']]
            fig_status = px.scatter(
                x=X_pca[:, 0], y=X_pca[:, 1], 
                color=df['MonkeyPox'],
                title="Status Monkeypox Asli",
                labels={'x': f'Komponen 1 ({pca.explained_variance_ratio_[0]:.1%})', 
                       'y': f'Komponen 2 ({pca.explained_variance_ratio_[1]:.1%})'},
                color_discrete_map={'Positive': 'red', 'Negative': 'blue'}
            )
            st.plotly_chart(fig_status, use_container_width=True)
        
        if show_detailed:
            st.subheader("Analisis Fitur Penting")
            
            # Heatmap fitur cluster
            fitur_cluster = df_encoded.copy()
            fitur_cluster['Cluster'] = cluster_labels
            rata_rata_cluster = fitur_cluster.groupby('Cluster').mean()
            
            # Ambil fitur teratas untuk visualisasi
            fitur_variance = rata_rata_cluster.var().nlargest(8)
            fitur_terpilih = list(fitur_variance.index)
            
            fig_heatmap = px.imshow(
                rata_rata_cluster[fitur_terpilih].T,
                aspect="auto",
                title="Profil Fitur per Cluster",
                labels=dict(x="Cluster", y="Fitur", color="Nilai"),
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        if show_comparison:
            st.subheader("Perbandingan dengan Hierarchical Clustering")
            
            # Sample untuk efisiensi
            ukuran_sampel = min(100, len(X_scaled))
            idx_sampel = np.random.choice(len(X_scaled), ukuran_sampel, replace=False)
            data_sampel = X_scaled[idx_sampel]
            
            # Hierarchical clustering
            linked = linkage(data_sampel, method='ward')
            hierarchical_labels = fcluster(linked, k_terbaik, criterion='maxclust')
            
            # Perbandingan kualitas
            sil_kmeans = silhouette_score(data_sampel, cluster_labels[idx_sampel])
            sil_hierarchical = silhouette_score(data_sampel, hierarchical_labels)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("K-Means Silhouette", f"{sil_kmeans:.3f}")
            with col2:
                st.metric("Hierarchical Silhouette", f"{sil_hierarchical:.3f}")
    
    with tab5:
        st.header("💡 Interpretasi dan Rekomendasi")
        
        # Komposisi cluster
        komposisi_cluster = pd.DataFrame({
            'Cluster': range(k_terbaik),
            'Total_Pasien': [len(df_result[df_result['Cluster'] == i]) for i in range(k_terbaik)],
            'Kasus_Positif': [len(df_result[(df_result['Cluster'] == i) & (df_result['MonkeyPox'] == 'Positive')]) for i in range(k_terbaik)],
            'Kasus_Negatif': [len(df_result[(df_result['Cluster'] == i) & (df_result['MonkeyPox'] == 'Negative')]) for i in range(k_terbaik)]
        })
        
        komposisi_cluster['Persen_Positif'] = (komposisi_cluster['Kasus_Positif'] / 
                                              komposisi_cluster['Total_Pasien'] * 100).round(1)
        
        # Profil risiko
        st.subheader("🎯 Profil Risiko per Cluster")
        
        fig_risk = px.bar(
            komposisi_cluster, 
            x='Cluster', 
            y='Persen_Positif',
            title="Persentase Kasus Positif per Cluster",
            color='Persen_Positif',
            color_continuous_scale='RdYlGn_r',
            labels={'Persen_Positif': 'Persentase Positif (%)'}
        )
        
        # Tambahkan garis referensi
        fig_risk.add_hline(y=70, line_dash="dash", line_color="red", 
                          annotation_text="Risiko Tinggi (>70%)")
        fig_risk.add_hline(y=30, line_dash="dash", line_color="green", 
                          annotation_text="Risiko Rendah (<30%)")
        
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Interpretasi per cluster
        st.subheader("🔍 Interpretasi Setiap Cluster")
        
        for idx, row in komposisi_cluster.iterrows():
            cluster_id = row['Cluster']
            total = row['Total_Pasien']
            persen_positif = row['Persen_Positif']
            
            if persen_positif > 70:
                risk_level = "🔴 RISIKO TINGGI"
                risk_color = "red"
            elif persen_positif < 30:
                risk_level = "🟢 RISIKO RENDAH"
                risk_color = "green"
            else:
                risk_level = "🟡 RISIKO SEDANG"
                risk_color = "orange"
            
            with st.container():
                st.markdown(f"""
                <div style="border-left: 4px solid {risk_color}; padding-left: 1rem; margin: 1rem 0;">
                    <h4>Cluster {cluster_id} - {risk_level}</h4>
                    <p><strong>Total Pasien:</strong> {total}</p>
                    <p><strong>Kasus Positif:</strong> {row['Kasus_Positif']} ({persen_positif}%)</p>
                    <p><strong>Kasus Negatif:</strong> {row['Kasus_Negatif']} ({100-persen_positif:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Kesimpulan dan Rekomendasi
        st.subheader("📋 Kesimpulan dan Rekomendasi")
        
        silhouette_final = silhouette_score(X_scaled, cluster_labels)
        
        # Kualitas clustering
        if silhouette_final > 0.5:
            quality_text = "sangat baik ✅"
        elif silhouette_final > 0.3:
            quality_text = "cukup baik ⚠️"
        else:
            quality_text = "perlu diperbaiki ❌"
        
        st.success(f"**Kualitas Clustering:** {quality_text} (Silhouette Score: {silhouette_final:.3f})")
        
        # Rekomendasi
        cluster_risiko_tinggi = komposisi_cluster[komposisi_cluster['Persen_Positif'] > 70]
        cluster_risiko_rendah = komposisi_cluster[komposisi_cluster['Persen_Positif'] < 30]
        
        st.markdown("### 🎯 Rekomendasi:")
        
        if len(cluster_risiko_tinggi) > 0:
            st.error(f"🔴 Ditemukan {len(cluster_risiko_tinggi)} cluster berisiko tinggi - Fokuskan perhatian medis")
        
        if len(cluster_risiko_rendah) > 0:
            st.success(f"🟢 Ditemukan {len(cluster_risiko_rendah)} cluster berisiko rendah - Dapat dijadikan grup kontrol")
        
        st.info("""
        **Saran Tindak Lanjut:**
        - Gunakan hasil clustering untuk stratifikasi risiko pasien
        - Validasi hasil dengan tim medis ahli
        - Pertimbangkan faktor klinis tambahan
        - Implementasikan monitoring khusus untuk cluster berisiko tinggi
        """)
        
        # Download hasil
        st.subheader("📥 Download Hasil")
        
        # Prepare download data
        hasil_clustering = df_result[['Patient_ID', 'MonkeyPox', 'Cluster']].copy()
        hasil_clustering['Risk_Level'] = hasil_clustering['Cluster'].apply(
            lambda x: 'High' if komposisi_cluster.iloc[x]['Persen_Positif'] > 70 
            else 'Low' if komposisi_cluster.iloc[x]['Persen_Positif'] < 30 
            else 'Medium'
        )
        
        csv_hasil = hasil_clustering.to_csv(index=False)
        st.download_button(
            label="📊 Download Hasil Clustering (CSV)",
            data=csv_hasil,
            file_name="hasil_clustering_monkeypox.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Silakan upload file dataset MonkeyPox.csv untuk memulai analisis")
    
    st.markdown("""
    ### 📋 Format File yang Dibutuhkan:
    - File CSV dengan kolom Patient_ID dan MonkeyPox
    - Kolom MonkeyPox berisi nilai 'Positive' atau 'Negative'
    - Kolom lainnya berisi fitur-fitur pasien untuk analisis clustering
    
    ### 🎯 Fitur Aplikasi:
    - Analisis clustering otomatis dengan K-Means
    - Pemilihan jumlah cluster optimal
    - Visualisasi interaktif dengan Plotly
    - Interpretasi risiko per cluster
    - Download hasil analisis
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🔬 Aplikasi Analisis Clustering Monkeypox | Dikembangkan dengan Streamlit</p>
</div>
""", unsafe_allow_html=True)