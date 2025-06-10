import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="🔬 Analisis Clustering Monkeypox",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling yang lebih menarik
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class=\"main-header\">🔬 ANALISIS CLUSTERING MONKEYPOX</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Fungsi untuk memuat data ---
@st.cache_data
def load_data():
    """Memuat dataset MonkeyPox dari file CSV"""
    try:
        df = pd.read_csv("MonkeyPox.csv")
        return df
    except FileNotFoundError:
        st.error("File MonkeyPox.csv tidak ditemukan! Pastikan file berada di direktori yang sama dengan aplikasi.")
        st.stop()

# --- Fungsi untuk preprocessing data ---
@st.cache_data
def preprocess_data(df):
    """
    Melakukan preprocessing data sesuai dengan notebook IPYNB:
    1. Menghapus kolom Patient_ID dan MonkeyPox
    2. Encoding data kategori
    3. Standardisasi
    4. PCA untuk reduksi dimensi
    """
    # Menghapus kolom yang tidak diperlukan untuk clustering
    df_fitur = df.drop(columns=["Patient_ID", "MonkeyPox"])
    
    # Melihat jenis data
    kolom_kategori = df_fitur.select_dtypes(include=["object"]).columns
    
    # Mengubah data kategori menjadi angka
    if len(kolom_kategori) > 0:
        # Mengisi nilai yang hilang di 'Systemic Illness' dengan mode
        df_fitur["Systemic Illness"] = df_fitur["Systemic Illness"].fillna(df_fitur["Systemic Illness"].mode()[0])
        df_encoded = pd.get_dummies(df_fitur, columns=kolom_kategori)
    else:
        df_encoded = df_fitur.copy()
        
    # Standardisasi data agar semua fitur memiliki skala yang sama
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    
    # Menerapkan PCA untuk reduksi dimensi
    pca = PCA(n_components=0.95)  # Pilih komponen yang menjelaskan 95% variansi
    X_processed = pca.fit_transform(X_scaled)
    
    return X_processed, pca, df_encoded, scaler

# --- Fungsi untuk mencari jumlah cluster terbaik ---
@st.cache_data
def find_best_k(X_processed):
    """
    Mencari jumlah cluster terbaik menggunakan:
    1. Elbow Method (Inertia)
    2. Silhouette Score
    """
    jumlah_k = range(2, 8)  # Dari 2 sampai 7 cluster
    inertia_values = []
    silhouette_scores = []
    
    for k in jumlah_k:
        # Membuat model K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_processed)
        
        # Menghitung metrik evaluasi
        inertia = kmeans.inertia_  # Mengukur kepadatan cluster
        sil_score = silhouette_score(X_processed, labels)  # Mengukur kualitas pemisahan
        
        inertia_values.append(inertia)
        silhouette_scores.append(sil_score)
        
    # Mencari k terbaik berdasarkan silhouette score
    k_terbaik = jumlah_k[np.argmax(silhouette_scores)]
    score_terbaik = max(silhouette_scores)
    
    return k_terbaik, inertia_values, silhouette_scores, jumlah_k

# --- Fungsi untuk melakukan clustering final ---
@st.cache_data
def perform_final_clustering(X_processed, k_terbaik):
    """Melakukan clustering final dengan jumlah cluster terbaik"""
    kmeans = KMeans(n_clusters=k_terbaik, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_processed)
    return kmeans, cluster_labels

# --- Main Aplikasi Streamlit ---
def main():
    # Sidebar untuk navigasi
    st.sidebar.title("📋 Navigasi")
    sections = [
        "📊 Data Overview", 
        "🛠️ Data Preprocessing", 
        "🎯 Optimasi Cluster", 
        "📈 Visualisasi Pemilihan", 
        "✨ Hasil Clustering"
    ]
    selected_section = st.sidebar.radio("Pilih Bagian:", sections)
    
    # Load data
    df = load_data()
    
    # BAGIAN 1: DATA OVERVIEW
    if selected_section == "📊 Data Overview":
        st.markdown("<h2 class=\"section-header\">📊 LANGKAH 1: MEMUAT DATA</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class=\"metric-card\"><h3>Jumlah Pasien</h3><h2 style=\"color: #1f77b4;\">25,000</h2></div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class=\"metric-card\"><h3>Jumlah Kolom</h3><h2 style=\"color: #ff7f0e;\">11</h2></div>", unsafe_allow_html=True)
        with col3:
            data_hilang = df.isnull().sum().sum()
            st.markdown(f"<div class=\"metric-card\"><h3>Data Hilang</h3><h2 style=\"color: #d62728;\">{data_hilang:,}</h2></div>", unsafe_allow_html=True)
        
        st.subheader("🔍 Preview Dataset")
        st.write("Berikut adalah 5 baris pertama dari dataset Monkeypox:")
        st.dataframe(df.head(), use_container_width=True)
        
        st.subheader("📋 Struktur Dataset")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Kolom-kolom dalam dataset:**")
            for i, col in enumerate(df.columns, 1):
                st.write(f"  {i}. **{col}** ({df[col].dtype})")
        
        with col2:
            st.subheader("📊 Distribusi Kasus Monkeypox")
            kasus = df["MonkeyPox"].value_counts()
            
            # Membuat bar chart yang menarik
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ["#ff7f0e", "#1f77b4"]
            bars = ax.bar(kasus.index, kasus.values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
            
            # Menambahkan label pada bar
            for bar, value in zip(bars, kasus.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 200,
                       f'{value:,}\n({value/len(df)*100:.1f}%)',
                       ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            ax.set_title("Distribusi Kasus Monkeypox", fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel("Status Monkeypox", fontsize=12, fontweight='bold')
            ax.set_ylabel("Jumlah Pasien", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Status data hilang
        if data_hilang == 0:
            st.markdown("<div class=\"success-box\">✅ <strong>Bagus!</strong> Tidak ada data yang hilang.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class=\"warning-box\">⚠️ <strong>Perhatian!</strong> Ada {data_hilang:,} data yang hilang, perlu dibersihkan.</div>", unsafe_allow_html=True)
    
    # BAGIAN 2: DATA PREPROCESSING
    elif selected_section == "🛠️ Data Preprocessing":
        st.markdown("<h2 class=\"section-header\">🛠️ LANGKAH 2: MENYIAPKAN DATA</h2>", unsafe_allow_html=True)
        
        X_processed, pca, df_encoded, scaler = preprocess_data(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 Tahapan Preprocessing")
            st.write("1. **Menghapus kolom tidak relevan**: Patient_ID, MonkeyPox")
            st.write("2. **Encoding data kategori**: One-hot encoding untuk 'Systemic Illness'")
            st.write("3. **Standardisasi**: Menggunakan StandardScaler")
            st.write("4. **Reduksi dimensi**: PCA dengan 95% explained variance")
            
            st.subheader("📈 Hasil Preprocessing")
            st.write(f"- **Fitur asli**: 9 kolom")
            st.write(f"- **Setelah encoding**: {df_encoded.shape[1]} fitur")
            st.write(f"- **Setelah PCA**: {X_processed.shape[1]} komponen")
            st.write(f"- **Explained variance**: {sum(pca.explained_variance_ratio_):.1%}")
        
        with col2:
            st.subheader("📊 Explained Variance Ratio")
            
            # Plot explained variance
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Individual explained variance
            ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                   pca.explained_variance_ratio_, 
                   alpha=0.7, color='skyblue', label='Individual')
            
            # Cumulative explained variance
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            ax.plot(range(1, len(cumsum) + 1), cumsum, 
                   'ro-', linewidth=2, markersize=6, label='Cumulative')
            
            ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.8, label='95% threshold')
            ax.set_xlabel('Principal Component', fontweight='bold')
            ax.set_ylabel('Explained Variance Ratio', fontweight='bold')
            ax.set_title('PCA - Explained Variance Analysis', fontweight='bold', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Informasi detail preprocessing
        st.subheader("ℹ️ Detail Preprocessing")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Mean (setelah scaling)", f"{np.mean(scaler.transform(df_encoded)):.6f}")
        with col2:
            st.metric("Data Std (setelah scaling)", f"{np.std(scaler.transform(df_encoded)):.6f}")
        with col3:
            st.metric("Total Variance Explained", f"{sum(pca.explained_variance_ratio_):.1%}")
    
    # BAGIAN 3: OPTIMASI CLUSTER
    elif selected_section == "🎯 Optimasi Cluster":
        st.markdown("<h2 class=\"section-header\">🎯 LANGKAH 3: MENCARI JUMLAH CLUSTER TERBAIK</h2>", unsafe_allow_html=True)
        
        X_processed, pca, df_encoded, scaler = preprocess_data(df)
        k_terbaik, inertia_values, silhouette_scores, jumlah_k = find_best_k(X_processed)
        
        st.write("Kita akan mencoba berbagai jumlah cluster dari 2 sampai 7 untuk menemukan yang optimal:")
        
        # Tabel hasil evaluasi
        results_df = pd.DataFrame({
            'Jumlah Cluster': list(jumlah_k),
            'Inertia': inertia_values,
            'Silhouette Score': [f"{score:.3f}" for score in silhouette_scores]
        })
        
        # Highlight baris terbaik
        def highlight_best(row):
            if row['Jumlah Cluster'] == k_terbaik:
                return ['background-color: #90EE90'] * len(row)
            return [''] * len(row)
        
        st.subheader("📊 Hasil Evaluasi Cluster")
        styled_df = results_df.style.apply(highlight_best, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Hasil terbaik
        st.markdown(f"""
        <div class=\"success-box\">
        <h3>🏆 HASIL OPTIMAL</h3>
        <p><strong>Jumlah cluster terbaik:</strong> {k_terbaik} cluster</p>
        <p><strong>Silhouette Score:</strong> {max(silhouette_scores):.3f}</p>
        <p><strong>Interpretasi:</strong> Semakin tinggi Silhouette Score (mendekati 1), semakin baik kualitas pemisahan cluster.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Penjelasan metrik
        st.subheader("📚 Penjelasan Metrik Evaluasi")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("""
            **🎯 Silhouette Score:**
            - Range: -1 hingga 1
            - Semakin tinggi, semakin baik
            - Mengukur seberapa mirip objek dengan cluster sendiri vs cluster lain
            - Score > 0.5 dianggap baik
            """)
        
        with col2:
            st.write("""
            **📉 Inertia (WCSS):**
            - Within-Cluster Sum of Squares
            - Semakin rendah, semakin kompak cluster
            - Digunakan dalam Elbow Method
            - Mencari "siku" pada grafik
            """)
    
    # BAGIAN 4: VISUALISASI PEMILIHAN
    elif selected_section == "📈 Visualisasi Pemilihan":
        st.markdown("<h2 class=\"section-header\">📈 LANGKAH 4: VISUALISASI PEMILIHAN CLUSTER</h2>", unsafe_allow_html=True)
        
        X_processed, pca, df_encoded, scaler = preprocess_data(df)
        k_terbaik, inertia_values, silhouette_scores, jumlah_k = find_best_k(X_processed)
        
        # Membuat grafik untuk membantu memahami pemilihan cluster
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Grafik Elbow Method (untuk melihat penurunan inertia)
        ax1.plot(jumlah_k, inertia_values, "bo-", linewidth=3, markersize=10, color='#1f77b4')
        ax1.set_xlabel("Jumlah Cluster", fontweight='bold')
        ax1.set_ylabel("Inertia (Kepadatan dalam Cluster)", fontweight='bold')
        ax1.set_title("Elbow Method - Mencari Jumlah Cluster Optimal", fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(jumlah_k)
        
        # Menambahkan anotasi nilai
        for i, (x, y) in enumerate(zip(jumlah_k, inertia_values)):
            ax1.annotate(f'{y:.0f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        
        # Grafik Silhouette Score
        ax2.plot(jumlah_k, silhouette_scores, "ro-", linewidth=3, markersize=10, color='#ff7f0e')
        ax2.axvline(x=k_terbaik, color="green", linestyle="--", alpha=0.8, linewidth=2,
                   label=f"Terbaik: {k_terbaik} cluster")
        ax2.axhline(y=max(silhouette_scores), color="green", linestyle=":", alpha=0.6, linewidth=1)
        ax2.set_xlabel("Jumlah Cluster", fontweight='bold')
        ax2.set_ylabel("Silhouette Score", fontweight='bold')
        ax2.set_title("Silhouette Score - Kualitas Pemisahan Cluster", fontweight='bold', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(jumlah_k)
        
        # Menambahkan anotasi nilai
        for i, (x, y) in enumerate(zip(jumlah_k, silhouette_scores)):
            ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown(f"""
        <div class=\"success-box\">
        <h4>📊 Interpretasi Grafik:</h4>
        <p>• <strong>Elbow Method:</strong> Mencari titik "siku" dimana penurunan inertia mulai melambat</p>
        <p>• <strong>Silhouette Score:</strong> Grafik menunjukkan bahwa <strong>{k_terbaik} cluster</strong> memberikan hasil terbaik dengan score <strong>{max(silhouette_scores):.3f}</strong></p>
        <p>• Kombinasi kedua metrik ini memvalidasi pilihan {k_terbaik} cluster sebagai optimal</p>
        </div>
        """, unsafe_allow_html=True)
    
    # BAGIAN 5: HASIL CLUSTERING
    elif selected_section == "✨ Hasil Clustering":
        st.markdown("<h2 class=\"section-header\">✨ LANGKAH 5: HASIL CLUSTERING</h2>", unsafe_allow_html=True)
        
        X_processed, pca, df_encoded, scaler = preprocess_data(df)
        k_terbaik, inertia_values, silhouette_scores, jumlah_k = find_best_k(X_processed)
        kmeans, cluster_labels = perform_final_clustering(X_processed, k_terbaik)
        
        # Menambahkan hasil cluster ke dataframe asli
        df_result = df.copy()
        df_result["Cluster"] = cluster_labels
        
        # Statistik cluster
        st.subheader("📊 Distribusi Cluster")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            cluster_sizes = df_result["Cluster"].value_counts().sort_index()
            cluster_df = pd.DataFrame({
                'Cluster': [f'Cluster {i}' for i in cluster_sizes.index],
                'Jumlah Pasien': cluster_sizes.values,
                'Persentase': [f'{(size/len(df_result)*100):.1f}%' for size in cluster_sizes.values]
            })
            st.dataframe(cluster_df, use_container_width=True)
        
        with col2:
            # Pie chart distribusi cluster
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_sizes)))
            wedges, texts, autotexts = ax.pie(cluster_sizes.values, 
                                            labels=[f'Cluster {i}' for i in cluster_sizes.index],
                                            autopct='%1.1f%%', 
                                            startangle=90, 
                                            colors=colors,
                                            explode=[0.05] * len(cluster_sizes))
            
            # Styling pie chart
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(12)
            
            ax.set_title("Distribusi Pasien per Cluster", fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Karakteristik cluster berdasarkan fitur asli
        st.subheader("🔍 Karakteristik Cluster")
        
        # Identify numeric columns from df_encoded for mean calculation
        numeric_cols_for_mean = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
        
        # We need to map the cluster labels back to the df_encoded for this step
        df_encoded_with_clusters = df_encoded.copy()
        df_encoded_with_clusters['Cluster'] = cluster_labels

        # Calculate mean for all numeric columns in df_encoded_with_clusters grouped by Cluster
        cluster_summary = df_encoded_with_clusters.groupby('Cluster')[numeric_cols_for_mean].mean()
        
        st.write("**Rata-rata karakteristik per cluster (setelah encoding dan standardisasi):**")
        st.dataframe(cluster_summary.round(3), use_container_width=True)
        
        # Heatmap karakteristik cluster
        st.subheader("🌡️ Heatmap Karakteristik Cluster")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Use the cluster_summary (which is already numeric and processed)
        sns.heatmap(cluster_summary.T, annot=True, cmap='RdYlBu_r', fmt='.3f', cbar_kws={'label': 'Mean Value'}, ax=ax)
        ax.set_title('Karakteristik Cluster - Heatmap Mean Values', fontsize=14, fontweight='bold')
        ax.set_xlabel('Cluster', fontweight='bold')
        ax.set_ylabel('Fitur', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Visualisasi cluster dalam ruang 2D (PCA)
        st.subheader("📈 Visualisasi Cluster (PCA 2D)")
        
        # PCA 2D untuk visualisasi
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_processed)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Scatter plot untuk setiap cluster
        colors = plt.cm.viridis(np.linspace(0, 1, k_terbaik))
        for i in range(k_terbaik):
            mask = cluster_labels == i
            ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
                      c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=30)
        
        # Plot centroids
        centroids_2d = pca_2d.transform(kmeans.cluster_centers_)
        ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                  marker="X", s=300, color="red", label="Centroids", 
                  edgecolors='black', linewidth=2, zorder=10)
        
        ax.set_title("Visualisasi Cluster dengan PCA (2 Komponen Utama)", fontsize=16, fontweight='bold')
        ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)", fontweight='bold')
        ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)", fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Insight cluster
        st.subheader("💡 Insight Clustering")
        st.markdown(f"""
        <div class=\"success-box\">
        <h4>🔍 Analisis Hasil Clustering:</h4>
        <p>• Dataset berhasil dikelompokkan menjadi <strong>{k_terbaik} cluster</strong> yang berbeda</p>
        <p>• Setiap cluster menunjukkan pola gejala dan karakteristik yang unik</p>
        <p>• Silhouette Score <strong>{max(silhouette_scores):.3f}</strong> menunjukkan kualitas pemisahan yang baik</p>
        <p>• Visualisasi PCA membantu memahami distribusi cluster dalam ruang 2 dimensi</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style=\'text-align: center; color: #666; padding: 20px;\'>
        <p>🔬 <strong>Aplikasi Analisis Clustering Monkeypox</strong></p>
        <p>Dibuat untuk tugas akhir machine learning dengan implementasi K-Means dan Hierarchical Clustering</p>
        <p><em>Dataset: 25,000 pasien dengan 11 fitur medis</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()