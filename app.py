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

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Analisis Clustering Mpox",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling yang lebih menarik
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
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
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat dan memproses data
@st.cache_data
def load_data():
    """
    Memuat dataset mpox dari file CSV dan melakukan preprocessing awal.
    Fungsi ini menggunakan caching untuk meningkatkan performa aplikasi.
    """
    try:
        # Mencoba memuat data dari file lokal
        df = pd.read_csv("mpox cases by country as of 30 June 2024.csv")
        # Mengubah kolom 'date' menjadi datetime
        df['date'] = pd.to_datetime(df['date'])
        # Mengambil data terbaru untuk setiap negara
        df = df.sort_values(by='date').drop_duplicates(subset=['country'], keep='last')
        return df
    except FileNotFoundError:
        st.error("Dataset tidak ditemukan! Pastikan file 'mpox cases by country as of 30 June 2024.csv' ada di direktori yang sama dengan app.py")
        return None

def preprocess_data(df, pca_variance=0.95):
    """
    Melakukan preprocessing data untuk clustering, termasuk encoding kategori,
    standardisasi, dan reduksi dimensi menggunakan PCA.
    
    Args:
        df: DataFrame input
        pca_variance: Proporsi varians yang ingin dipertahankan dalam PCA
    
    Returns:
        Tuple berisi data yang telah diproses dan objek-objek preprocessing
    """
    # Menghapus kolom identifier
    df_features = df.drop(columns=["country", "date", "iso3", "who_region"])
    
    # Identifikasi kolom kategori dan numerik
    categorical_cols = df_features.select_dtypes(include=["object"]).columns
    numerical_cols = df_features.select_dtypes(include=[np.number]).columns
    
    # One-hot encoding untuk kolom kategori
    if len(categorical_cols) > 0:
        df_encoded = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)
    else:
        df_encoded = df_features.copy()
    
    # Standardisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    
    # PCA untuk reduksi dimensi
    pca = PCA(n_components=pca_variance)
    X_processed = pca.fit_transform(X_scaled)
    
    return X_processed, X_scaled, df_encoded, scaler, pca, categorical_cols, numerical_cols

def find_optimal_clusters(X, max_clusters=10):
    """
    Mencari jumlah cluster optimal menggunakan metode Elbow dan Silhouette Score.
    
    Args:
        X: Data yang telah diproses
        max_clusters: Jumlah maksimum cluster yang akan diuji
    
    Returns:
        Tuple berisi rentang k, inertia values, dan silhouette scores
    """
    k_range = range(2, max_clusters + 1)
    inertia_values = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        inertia_values.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
    
    return k_range, inertia_values, silhouette_scores

def perform_clustering(X, n_clusters, random_state=42):
    """
    Melakukan clustering K-Means dengan parameter yang ditentukan.
    
    Args:
        X: Data yang telah diproses
        n_clusters: Jumlah cluster yang diinginkan
        random_state: Seed untuk reproducibility
    
    Returns:
        Tuple berisi model KMeans dan label cluster
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    return kmeans, labels

# Header utama aplikasi
st.markdown("<h1 class=\"main-header\">🔬 Analisis Clustering Mpox Kasus Per Negara</h1>", unsafe_allow_html=True)

# Sidebar untuk kontrol parameter
st.sidebar.markdown("## ⚙️ Pengaturan Clustering")
st.sidebar.markdown("Gunakan panel ini untuk menyesuaikan parameter analisis clustering")

# Memuat data
df = load_data()

if df is not None:
    # Menampilkan informasi dataset di sidebar
    st.sidebar.markdown("### 📊 Informasi Dataset")
    st.sidebar.info(f"""
    **Jumlah Negara:** {df.shape[0]}
    **Jumlah Fitur:** {df.shape[1]}
    **Periode Data:** Hingga 30 Juni 2024
    """)
    
    # Parameter kontrol di sidebar
    st.sidebar.markdown("### 🎛️ Parameter PCA")
    pca_variance = st.sidebar.slider(
        "Proporsi Varians yang Dipertahankan",
        min_value=0.80,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help="Menentukan berapa persen varians data yang ingin dipertahankan setelah reduksi dimensi"
    )
    
    st.sidebar.markdown("### 🎯 Parameter Clustering")
    max_clusters_test = st.sidebar.slider(
        "Maksimum Cluster untuk Pengujian",
        min_value=5,
        max_value=15,
        value=10,
        help="Menentukan rentang jumlah cluster yang akan diuji untuk mencari nilai optimal"
    )
    
    # Checkbox untuk clustering otomatis atau manual
    auto_clustering = st.sidebar.checkbox(
        "Clustering Otomatis",
        value=True,
        help="Jika dicentang, sistem akan otomatis memilih jumlah cluster terbaik berdasarkan Silhouette Score"
    )
    
    if not auto_clustering:
        manual_clusters = st.sidebar.slider(
            "Jumlah Cluster Manual",
            min_value=2,
            max_value=max_clusters_test,
            value=3,
            help="Pilih jumlah cluster secara manual"
        )
    
    # Tombol untuk menjalankan analisis
    if st.sidebar.button("🚀 Jalankan Analisis", type="primary"):
        
        # Progress bar untuk menunjukkan kemajuan analisis
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Langkah 1: Preprocessing data
        status_text.text("Memproses data...")
        progress_bar.progress(20)
        
        # Menghapus kolom 'date', 'iso3', 'who_region' dari df_features
        X_processed, X_scaled, df_encoded, scaler, pca, categorical_cols, numerical_cols = preprocess_data(df.copy(), pca_variance)
        
        # Langkah 2: Mencari cluster optimal
        status_text.text("Mencari jumlah cluster optimal...")
        progress_bar.progress(40)
        
        k_range, inertia_values, silhouette_scores = find_optimal_clusters(X_processed, max_clusters_test)
        
        # Menentukan jumlah cluster
        if auto_clustering:
            optimal_k = k_range[np.argmax(silhouette_scores)]
            optimal_score = max(silhouette_scores)
        else:
            optimal_k = manual_clusters
            # Mencari silhouette score untuk cluster manual
            kmeans_temp = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            labels_temp = kmeans_temp.fit_predict(X_processed)
            optimal_score = silhouette_score(X_processed, labels_temp)
        
        # Langkah 3: Melakukan clustering final
        status_text.text("Melakukan clustering final...")
        progress_bar.progress(60)
        
        kmeans_final, cluster_labels = perform_clustering(X_processed, optimal_k)
        
        # Menambahkan hasil clustering ke dataframe
        df["Cluster"] = cluster_labels
        
        # Langkah 4: Mempersiapkan visualisasi
        status_text.text("Mempersiapkan visualisasi...")
        progress_bar.progress(80)
        
        # PCA 2D untuk visualisasi
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_scaled)
        
        # Clustering untuk visualisasi 2D
        kmeans_2d, cluster_labels_2d = perform_clustering(X_pca_2d, optimal_k)
        
        progress_bar.progress(100)
        status_text.text("Analisis selesai!")
        
        # Menyimpan hasil dalam session state untuk persistensi
        st.session_state.analysis_complete = True
        st.session_state.df_clustered = df.copy()
        st.session_state.optimal_k = optimal_k
        st.session_state.optimal_score = optimal_score
        st.session_state.k_range = k_range
        st.session_state.inertia_values = inertia_values
        st.session_state.silhouette_scores = silhouette_scores
        st.session_state.X_pca_2d = X_pca_2d
        st.session_state.cluster_labels_2d = cluster_labels_2d
        st.session_state.pca_2d = pca_2d
        st.session_state.numerical_cols = numerical_cols
        st.session_state.categorical_cols = categorical_cols
        st.session_state.pca_variance_used = sum(pca.explained_variance_ratio_)
        
        # Menghapus progress bar dan status text
        progress_bar.empty()
        status_text.empty()

# Menampilkan hasil analisis jika sudah selesai
if hasattr(st.session_state, 'analysis_complete') and st.session_state.analysis_complete:
    
    # Mengambil data dari session state
    df_clustered = st.session_state.df_clustered
    optimal_k = st.session_state.optimal_k
    optimal_score = st.session_state.optimal_score
    k_range = st.session_state.k_range
    inertia_values = st.session_state.inertia_values
    silhouette_scores = st.session_state.silhouette_scores
    X_pca_2d = st.session_state.X_pca_2d
    cluster_labels_2d = st.session_state.cluster_labels_2d
    pca_2d = st.session_state.pca_2d
    numerical_cols = st.session_state.numerical_cols
    pca_variance_used = st.session_state.pca_variance_used
    
    # Menampilkan ringkasan hasil
    st.markdown("<div class=\"section-header\">📊 Ringkasan Hasil Clustering</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Jumlah Cluster Optimal",
            value=optimal_k,
            help="Jumlah cluster yang memberikan hasil terbaik berdasarkan Silhouette Score"
        )
    
    with col2:
        st.metric(
            label="Silhouette Score",
            value=f"{optimal_score:.3f}",
            help="Skor kualitas clustering (semakin mendekati 1, semakin baik)"
        )
    
    with col3:
        st.metric(
            label="Varians PCA",
            value=f"{pca_variance_used:.1%}",
            help="Persentase varians data yang dipertahankan setelah reduksi dimensi"
        )
    
    with col4:
        st.metric(
            label="Total Negara",
            value=len(df_clustered),
            help="Jumlah total negara yang dianalisis"
        )
    
    # Tab untuk berbagai analisis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Pemilihan Cluster", 
        "🌍 Visualisasi 2D", 
        "📈 Distribusi Cluster",
        "🔍 Profil Cluster",
        "📋 Data Detail"
    ])
    
    with tab1:
        st.markdown("<div class=\"section-header\">Analisis Pemilihan Jumlah Cluster</div>", unsafe_allow_html=True)
        
        # Membuat subplot untuk Elbow Method dan Silhouette Score
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Elbow Method', 'Silhouette Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Elbow Method
        fig.add_trace(
            go.Scatter(
                x=list(k_range), 
                y=inertia_values,
                mode='lines+markers',
                name='Inertia',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Silhouette Score
        fig.add_trace(
            go.Scatter(
                x=list(k_range), 
                y=silhouette_scores,
                mode='lines+markers',
                name='Silhouette Score',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ),
            row=1, col=2
        )
        
        # Menandai cluster optimal
        fig.add_vline(
            x=optimal_k, 
            line_dash="dash", 
            line_color="green",
            annotation_text=f"Optimal: {optimal_k}",
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            title_text="Analisis Pemilihan Jumlah Cluster",
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Jumlah Cluster (k)", row=1, col=1)
        fig.update_xaxes(title_text="Jumlah Cluster (k)", row=1, col=2)
        fig.update_yaxes(title_text="Inertia", row=1, col=1)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretasi hasil
        st.markdown("<div class=\"info-box\">", unsafe_allow_html=True)
        st.markdown(f"""
        **Interpretasi Hasil:**
        
        **Elbow Method** membantu kita mengidentifikasi titik di mana penambahan cluster tidak memberikan penurunan inertia yang signifikan lagi. Pada grafik ini, kita dapat melihat "siku" atau titik belok yang menunjukkan jumlah cluster yang efisien.
        
        **Silhouette Score** mengukur seberapa baik setiap data point cocok dengan cluster-nya dibandingkan dengan cluster lain. Skor berkisar dari -1 hingga 1, di mana:
        - Nilai mendekati 1: Data point sangat cocok dengan cluster-nya
        - Nilai mendekati 0: Data point berada di perbatasan antar cluster
        - Nilai negatif: Data point mungkin ditempatkan di cluster yang salah
        
        Berdasarkan analisis, **{optimal_k} cluster** memberikan Silhouette Score terbaik sebesar **{optimal_score:.3f}**.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class=\"section-header\">Visualisasi Clustering 2D</div>", unsafe_allow_html=True)
        
        # Membuat DataFrame untuk visualisasi
        df_vis = pd.DataFrame({
            'PC1': X_pca_2d[:, 0],
            'PC2': X_pca_2d[:, 1],
            'Cluster': cluster_labels_2d,
            'Country': df_clustered['country'],
            'WHO_Region': df_clustered['who_region']
        })
        
        # Scatter plot interaktif
        fig = px.scatter(
            df_vis, 
            x='PC1', 
            y='PC2', 
            color='Cluster',
            hover_data=['Country', 'WHO_Region'],
            title=f'Hasil Clustering K-Means (Visualisasi 2D) - {optimal_k} Cluster',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title=f'Komponen 1 ({pca_2d.explained_variance_ratio_[0]:.1%} varians)',
            yaxis_title=f'Komponen 2 ({pca_2d.explained_variance_ratio_[1]:.1%} varians)',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretasi hasil
        st.markdown("<div class=\"info-box\">", unsafe_allow_html=True)
        st.markdown(f"""
        **Interpretasi Visualisasi 2D:**
        
        Visualisasi ini menggunakan **Principal Component Analysis (PCA)** untuk mereduksi data multidimensi menjadi 2 dimensi agar dapat ditampilkan dalam grafik. Setiap titik mewakili satu negara, dan warna menunjukkan cluster yang ditetapkan. Negara-negara yang berdekatan dalam grafik memiliki karakteristik Mpox yang serupa.
        
        - **Komponen 1** menjelaskan {pca_2d.explained_variance_ratio_[0]:.1%} dari total variasi dalam data.
        - **Komponen 2** menjelaskan {pca_2d.explained_variance_ratio_[1]:.1%} dari total variasi dalam data.
        - **Total variasi yang dijelaskan oleh 2 komponen ini:** {sum(pca_2d.explained_variance_ratio_):.1%}.
        
        Perhatikan bagaimana negara-negara dengan karakteristik serupa cenderung mengelompok bersama, membentuk cluster yang berbeda. Ini menunjukkan bahwa algoritma K-Means berhasil mengidentifikasi pola-pola tersembunyi dalam data.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<div class=\"section-header\">Distribusi dan Karakteristik Cluster</div>", unsafe_allow_html=True)
        
        # Distribusi negara per cluster
        cluster_distribution = df_clustered['Cluster'].value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart distribusi cluster
            fig_pie = px.pie(
                values=cluster_distribution.values,
                names=[f'Cluster {i}' for i in cluster_distribution.index],
                title='Distribusi Negara per Cluster'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown("""
            <div class="info-box">
            **Interpretasi Pie Chart Distribusi Cluster:**
            
            Pie chart ini menunjukkan proporsi negara yang termasuk dalam setiap cluster. Ukuran setiap irisan merepresentasikan jumlah negara dalam cluster tersebut. Visualisasi ini membantu kita memahami seberapa merata atau tidak merata distribusi negara di antara cluster-cluster yang terbentuk.
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Bar chart distribusi WHO Region per cluster
            fig_bar = px.histogram(
                df_clustered, 
                x='Cluster', 
                color='who_region',
                title='Distribusi WHO Region per Cluster',
                barmode='group'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("""
            <div class="info-box">
            **Interpretasi Bar Chart Distribusi WHO Region per Cluster:**
            
            Bar chart ini menampilkan komposisi WHO Region di setiap cluster. Setiap bar mewakili sebuah cluster, dan segmen warna di dalamnya menunjukkan jumlah negara dari masing-masing WHO Region. Ini membantu kita melihat apakah ada cluster yang didominasi oleh negara-negara dari WHO Region tertentu, yang bisa mengindikasikan pola geografis dalam penyebaran Mpox.
            </div>
            """, unsafe_allow_html=True)
        
        # Tabel distribusi detail
        st.markdown("### 📊 Distribusi Detail per Cluster")
        
        distribution_data = []
        for cluster_id in sorted(df_clustered['Cluster'].unique()):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
            distribution_data.append({
                'Cluster': cluster_id,
                'Jumlah Negara': len(cluster_data),
                'Persentase': f"{(len(cluster_data)/len(df_clustered)*100):.1f}%",
                'Negara Contoh': ', '.join(cluster_data['country'].head(3).tolist())
            })
        
        distribution_df = pd.DataFrame(distribution_data)
        st.dataframe(distribution_df, use_container_width=True)
        st.markdown("""
        <div class="info-box">
        **Interpretasi Tabel Distribusi Detail per Cluster:**
        
        Tabel ini memberikan rincian numerik tentang setiap cluster, termasuk jumlah negara, persentase dari total negara, dan beberapa contoh negara yang termasuk dalam cluster tersebut. Ini melengkapi visualisasi di atas dengan data konkret, memungkinkan analisis yang lebih mendalam tentang ukuran dan komposisi setiap cluster.
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("<div class=\"section-header\">Profil Karakteristik Cluster</div>", unsafe_allow_html=True)
        
        # Analisis rata-rata fitur numerik per cluster
        numeric_features = [col for col in numerical_cols if col in df_clustered.columns]
        
        if numeric_features:
            cluster_profiles = df_clustered.groupby('Cluster')[numeric_features].mean()
            
            # Heatmap profil cluster
            fig_heatmap = px.imshow(
                cluster_profiles.T,
                title='Profil Karakteristik Cluster (Rata-rata)',
                labels=dict(x="Cluster", y="Fitur", color="Nilai Rata-rata"),
                aspect="auto"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            st.markdown("""
            <div class="info-box">
            **Interpretasi Heatmap Profil Karakteristik Cluster:**
            
            Heatmap ini menampilkan nilai rata-rata dari fitur-fitur numerik untuk setiap cluster. Warna yang lebih gelap atau lebih terang menunjukkan nilai rata-rata yang lebih tinggi atau lebih rendah untuk fitur tertentu dalam cluster tersebut. Ini sangat berguna untuk mengidentifikasi fitur-fitur yang paling membedakan antar cluster, membantu kita memahami karakteristik unik dari setiap kelompok negara.
            </div>
            """, unsafe_allow_html=True)
            
            # Tabel profil cluster
            st.markdown("### 📋 Profil Numerik Detail per Cluster")
            st.dataframe(cluster_profiles, use_container_width=True)
            st.markdown("""
            <div class="info-box">
            **Interpretasi Tabel Profil Numerik Detail per Cluster:**
            
            Tabel ini menyajikan nilai rata-rata yang tepat untuk setiap fitur numerik di setiap cluster. Ini adalah data mentah yang digunakan untuk membuat heatmap di atas, memungkinkan pemeriksaan detail dan perbandingan antar cluster untuk setiap fitur secara individual.
            </div>
            """, unsafe_allow_html=True)
            
            # Box plots untuk fitur numerik
            st.markdown("### 📦 Distribusi Fitur Numerik per Cluster")
            
            selected_feature = st.selectbox(
                "Pilih fitur untuk analisis distribusi:",
                numeric_features,
                help="Pilih fitur numerik untuk melihat distribusinya pada setiap cluster"
            )
            
            fig_box = px.box(
                df_clustered, 
                x='Cluster', 
                y=selected_feature,
                title=f'Distribusi {selected_feature} per Cluster'
            )
            st.plotly_chart(fig_box, use_container_width=True)
            st.markdown(f"""
            <div class="info-box">
            **Interpretasi Box Plot Distribusi {selected_feature} per Cluster:**
            
            Box plot ini menunjukkan distribusi nilai untuk fitur '{selected_feature}' di setiap cluster. Box plot menampilkan median, kuartil, dan potensi outlier, memberikan gambaran tentang sebaran data dalam setiap cluster. Ini membantu kita memahami variabilitas dan perbedaan nilai fitur di antara cluster, serta mengidentifikasi cluster mana yang memiliki nilai ekstrem untuk fitur tertentu.
            </div>
            """, unsafe_allow_html=True)
            
            # Interpretasi cluster
            st.markdown("### 🔍 Interpretasi Cluster")
            
            for cluster_id in sorted(df_clustered['Cluster'].unique()):
                cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
                
                with st.expander(f"Cluster {cluster_id} ({len(cluster_data)} negara)"):
                    
                    # Statistik dasar
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Negara-negara dalam cluster:**")
                        countries_list = cluster_data['country'].tolist()
                        for i in range(0, len(countries_list), 3):
                            st.write(", ".join(countries_list[i:i+3]))
                    
                    with col2:
                        st.markdown("**Distribusi WHO Region:**")
                        region_counts = cluster_data['who_region'].value_counts()
                        for region, count in region_counts.items():
                            st.write(f"- {region}: {count} negara")
                    
                    # Karakteristik utama
                    st.markdown("**Karakteristik Utama:**")
                    cluster_profile = cluster_profiles.loc[cluster_id]
                    
                    # Mencari fitur dengan nilai tertinggi dan terendah
                    highest_feature = cluster_profile.idxmax()
                    lowest_feature = cluster_profile.idxmin()
                    
                    st.write(f"- Fitur tertinggi: {highest_feature} ({cluster_profile[highest_feature]:.2f})")
                    st.write(f"- Fitur terendah: {lowest_feature} ({cluster_profile[lowest_feature]:.2f})")
                    st.markdown("""
                    <div class="info-box">
                    **Interpretasi Detail Cluster:**
                    
                    Bagian ini memberikan rincian mendalam untuk setiap cluster, termasuk daftar negara anggotanya, distribusi regional, dan fitur-fitur numerik yang paling menonjol (tertinggi dan terendah). Informasi ini krusial untuk memberikan nama atau label yang bermakna pada setiap cluster, serta untuk memahami profil risiko atau karakteristik epidemiologi Mpox yang spesifik untuk kelompok negara tersebut.
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab5:
        st.markdown("<div class=\"section-header\">Data Detail dengan Hasil Clustering</div>", unsafe_allow_html=True)
        
        # Filter data berdasarkan cluster
        selected_clusters = st.multiselect(
            "Filter berdasarkan Cluster:",
            options=sorted(df_clustered['Cluster'].unique()),
            default=sorted(df_clustered['Cluster'].unique()),
            help="Pilih cluster yang ingin ditampilkan dalam tabel"
        )
        
        if selected_clusters:
            filtered_data = df_clustered[df_clustered['Cluster'].isin(selected_clusters)]
        else:
            filtered_data = df_clustered
        
        # Menampilkan tabel data
        st.dataframe(filtered_data, use_container_width=True)
        st.markdown("""
        <div class="info-box">
        **Interpretasi Tabel Data Detail:**
        
        Tabel ini menampilkan data mentah yang telah digabungkan dengan hasil clustering. Pengguna dapat memfilter data berdasarkan cluster yang dipilih untuk melihat negara-negara spesifik dan data terkait yang termasuk dalam cluster tersebut. Ini adalah alat yang berguna untuk eksplorasi data secara langsung dan memverifikasi hasil clustering.
        </div>
        """, unsafe_allow_html=True)
        
        # Statistik ringkasan
        st.markdown("### 📊 Statistik Ringkasan Data Terpilih")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Jumlah Negara", len(filtered_data))
        
        with col2:
            if 'total_confirmed_cases' in filtered_data.columns:
                st.metric("Total Kasus Terkonfirmasi", f"{filtered_data['total_confirmed_cases'].sum():,.0f}")
            else:
                st.metric("Total Kasus Terkonfirmasi", "N/A")
        
        with col3:
            if 'total_deaths' in filtered_data.columns:
                st.metric("Total Kematian", f"{filtered_data['total_deaths'].sum():,.0f}")
            else:
                st.metric("Total Kematian", "N/A")
        st.markdown("""
        <div class="info-box">
        **Interpretasi Statistik Ringkasan Data Terpilih:**
        
        Bagian ini menyajikan ringkasan statistik dari data yang saat ini ditampilkan dalam tabel detail. Ini mencakup jumlah negara, total kasus terkonfirmasi, dan total kematian. Statistik ini diperbarui secara dinamis berdasarkan filter cluster yang diterapkan, memungkinkan pengguna untuk dengan cepat mendapatkan gambaran umum tentang dampak Mpox dalam subset data yang diminati.
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("Silakan unggah dataset dan klik 'Jalankan Analisis' di sidebar untuk memulai.")


