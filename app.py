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
    df_fitur = df.drop(columns=[col for col in ["Patient_ID", "MonkeyPox"] if col in df.columns])
    
    # Melihat jenis data
    kolom_kategori = df_fitur.select_dtypes(include=["object"]).columns
    
    # Mengubah data kategori menjadi angka
    if len(kolom_kategori) > 0:
        # Mengisi nilai yang hilang di kolom kategori dengan mode
        for col in kolom_kategori:
            if df_fitur[col].isnull().any():
                df_fitur[col] = df_fitur[col].fillna(df_fitur[col].mode()[0])
        df_encoded = pd.get_dummies(df_fitur, columns=kolom_kategori)
    else:
        df_encoded = df_fitur.copy()
        # Jika ada kolom numeric dengan NaN, bisa di-handle di luar
    
    # Pastikan tidak ada NaN sebelum scaling
    if df_encoded.isnull().sum().sum() > 0:
        df_encoded = df_encoded.fillna(df_encoded.mean())
    
    # Standardisasi data agar semua fitur memiliki skala yang sama
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(df_encoded)
    except Exception as e:
        st.error(f"Error saat scaling data: {e}")
        st.stop()
    
    # Menerapkan PCA untuk reduksi dimensi
    try:
        pca = PCA(n_components=0.95, random_state=42)  # Pilih komponen yang menjelaskan 95% variansi
        X_processed = pca.fit_transform(X_scaled)
    except Exception as e:
        st.error(f"Error saat PCA: {e}")
        st.stop()
    
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
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_processed)
        inertia_values.append(kmeans.inertia_)
        try:
            sil_score = silhouette_score(X_processed, labels)
        except Exception:
            sil_score = np.nan
        silhouette_scores.append(sil_score)
    
    # Pilih k dengan silhouette highest, jika semua NaN, fallback ke elbow (k=2)
    if not all(np.isnan(silhouette_scores)):
        k_terbaik = jumlah_k[np.nanargmax(silhouette_scores)]
        score_terbaik = np.nanmax(silhouette_scores)
    else:
        k_terbaik = 2
        score_terbaik = silhouette_scores[0]
    
    return k_terbaik, inertia_values, silhouette_scores, list(jumlah_k)

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
            st.markdown("<div class=\"metric-card\"><h3>Jumlah Pasien</h3><h2 style=\"color: #1f77b4;\">{:,}</h2></div>".format(len(df)), unsafe_allow_html=True)
        with col2:
            st.markdown("<div class=\"metric-card\"><h3>Jumlah Kolom</h3><h2 style=\"color: #ff7f0e;\">{}</h2></div>".format(len(df.columns)), unsafe_allow_html=True)
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
            if 'MonkeyPox' in df.columns:
                st.subheader("📊 Distribusi Kasus Monkeypox")
                kasus = df["MonkeyPox"].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ["#ff7f0e", "#1f77b4"]
                bars = ax.bar(kasus.index.astype(str), kasus.values, color=colors[:len(kasus)], alpha=0.8, edgecolor='black', linewidth=1.2)
                for bar, value in zip(bars, kasus.values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(kasus.values)*0.01,
                           f'{value:,}\n({value/len(df)*100:.1f}%)',
                           ha='center', va='bottom', fontweight='bold', fontsize=11)
                ax.set_title("Distribusi Kasus Monkeypox", fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel("Status Monkeypox", fontsize=12, fontweight='bold')
                ax.set_ylabel("Jumlah Pasien", fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                st.pyplot(fig)
        
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
            st.write("1. **Menghapus kolom tidak relevan**: Patient_ID, MonkeyPox jika ada")
            st.write("2. **Encoding data kategori**: One-hot encoding untuk kolom kategori")
            st.write("3. **Standardisasi**: Menggunakan StandardScaler")
            st.write("4. **Reduksi dimensi**: PCA dengan 95% explained variance")
            st.subheader("📈 Hasil Preprocessing")
            st.write(f"- **Fitur asli**: {df_encoded.shape[1]} kolom setelah encoding")
            st.write(f"- **Setelah PCA**: {X_processed.shape[1]} komponen")
            st.write(f"- **Explained variance**: {sum(pca.explained_variance_ratio_):.1%}")
        with col2:
            st.subheader("📊 Explained Variance Ratio")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                   pca.explained_variance_ratio_, 
                   alpha=0.7, color='skyblue', label='Individual')
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
        st.subheader("ℹ️ Detail Preprocessing")
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                mean_val = np.mean(scaler.transform(df_encoded))
            except Exception:
                mean_val = np.nan
            st.metric("Data Mean (setelah scaling)", f"{mean_val:.6f}" if not np.isnan(mean_val) else "NaN")
        with col2:
            try:
                std_val = np.std(scaler.transform(df_encoded))
            except Exception:
                std_val = np.nan
            st.metric("Data Std (setelah scaling)", f"{std_val:.6f}" if not np.isnan(std_val) else "NaN")
        with col3:
            st.metric("Total Variance Explained", f"{sum(pca.explained_variance_ratio_):.1%}")
    
    # BAGIAN 3: OPTIMASI CLUSTER
    elif selected_section == "🎯 Optimasi Cluster":
        st.markdown("<h2 class=\"section-header\">🎯 LANGKAH 3: MENCARI JUMLAH CLUSTER TERBAIK</h2>", unsafe_allow_html=True)
        X_processed, pca, df_encoded, scaler = preprocess_data(df)
        k_terbaik, inertia_values, silhouette_scores, jumlah_k = find_best_k(X_processed)
        st.write("Kita akan mencoba berbagai jumlah cluster dari 2 sampai 7 untuk menemukan yang optimal:")
        results_df = pd.DataFrame({
            'Jumlah Cluster': jumlah_k,
            'Inertia': inertia_values,
            'Silhouette Score': [f"{score:.3f}" if not np.isnan(score) else 'nan' for score in silhouette_scores]
        })
        def highlight_best(row):
            try:
                if int(row['Jumlah Cluster']) == k_terbaik:
                    return ['background-color: #90EE90'] * len(row)
            except:
                pass
            return [''] * len(row)
        st.subheader("📊 Hasil Evaluasi Cluster")
        styled_df = results_df.style.apply(highlight_best, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        if not np.isnan(k_terbaik):
            st.markdown(f"""
            <div class=\"success-box\">
            <h3>🏆 HASIL OPTIMAL</h3>
            <p><strong>Jumlah cluster terbaik:</strong> {k_terbaik} cluster</p>
            <p><strong>Silhouette Score:</strong> {score_terbaik:.3f}</p>
            <p><strong>Interpretasi:</strong> Semakin tinggi Silhouette Score (mendekati 1), semakin baik kualitas pemisahan cluster.</p>
            </div>
            """, unsafe_allow_html=True)
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
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        ax1, ax2 = axes
        ax1.plot(jumlah_k, inertia_values, 'bo-', linewidth=3, markersize=10)
        ax1.set_xlabel("Jumlah Cluster", fontweight='bold')
        ax1.set_ylabel("Inertia (Kepadatan dalam Cluster)", fontweight='bold')
        ax1.set_title("Elbow Method - Mencari Jumlah Cluster Optimal", fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(jumlah_k)
        for x, y in zip(jumlah_k, inertia_values):
            ax1.annotate(f'{y:.0f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        ax2.plot(jumlah_k, silhouette_scores, 'ro-', linewidth=3, markersize=10)
        ax2.axvline(x=k_terbaik, color="green", linestyle="--", alpha=0.8, linewidth=2,
                   label=f"Terbaik: {k_terbaik} cluster")
        if not np.isnan(max(silhouette_scores)):
            ax2.axhline(y=max(silhouette_scores), color="green", linestyle=":", alpha=0.6, linewidth=1)
        ax2.set_xlabel("Jumlah Cluster", fontweight='bold')
        ax2.set_ylabel("Silhouette Score", fontweight='bold')
        ax2.set_title("Silhouette Score - Kualitas Pemisahan Cluster", fontweight='bold', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(jumlah_k)
        for x, y in zip(jumlah_k, silhouette_scores):
            label = f'{y:.3f}' if not np.isnan(y) else 'nan'
            ax2.annotate(label, (x, y if not np.isnan(y) else 0), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown(f"""
        <div class=\"success-box\">
        <h4>📊 Interpretasi Grafik:</h4>
        <p>• <strong>Elbow Method:</strong> Mencari titik "siku" dimana penurunan inertia mulai melambat</p>
        <p>• <strong>Silhouette Score:</strong> Grafik menunjukkan bahwa <strong>{k_terbaik} cluster</strong> memberikan hasil terbaik dengan score <strong>{max(silhouette_scores) if not np.isnan(max(silhouette_scores)) else 'nan'}:.3f</strong></p>
        <p>• Kombinasi kedua metrik ini memvalidasi pilihan {k_terbaik} cluster sebagai optimal</p>
        </div>
        """, unsafe_allow_html=True)
    
    # BAGIAN 5: HASIL CLUSTERING
    elif selected_section == "✨ Hasil Clustering":
        st.markdown("<h2 class=\"section-header\">✨ LANGKAH 5: HASIL CLUSTERING</h2>", unsafe_allow_html=True)
        X_processed, pca, df_encoded, scaler = preprocess_data(df)
        k_terbaik, inertia_values, silhouette_scores, jumlah_k = find_best_k(X_processed)
        kmeans, cluster_labels = perform_final_clustering(X_processed, k_terbaik)
        df_result = df.copy()
        df_result["Cluster"] = cluster_labels
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
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_sizes)))
            wedges, texts, autotexts = ax.pie(cluster_sizes.values, 
                                            labels=[f'Cluster {i}' for i in cluster_sizes.index],
                                            autopct='%1.1f%%', 
                                            startangle=90, 
                                            colors=colors,
                                            explode=[0.05] * len(cluster_sizes))
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(12)
            ax.set_title("Distribusi Pasien per Cluster", fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            st.pyplot(fig)
        # Karakteristik cluster berdasarkan data terstandarisasi
        st.subheader("🔍 Karakteristik Cluster")
        # Hitung mean pada data terstandarisasi
        try:
            df_scaled = pd.DataFrame(scaler.transform(df_encoded), columns=df_encoded.columns)
            df_scaled['Cluster'] = cluster_labels
            cluster_summary = df_scaled.groupby('Cluster').mean()
        except Exception as e:
            st.error(f"Gagal menghitung karakteristik cluster: {e}")
            cluster_summary = pd.DataFrame()
        if not cluster_summary.empty:
            st.write("**Rata-rata karakteristik per cluster (data terstandarisasi):**")
            st.dataframe(cluster_summary.round(3), use_container_width=True)
        else:
            st.warning("Karakteristik cluster tidak dapat dihitung (cluster_summary kosong).")
        st.subheader("🌡️ Heatmap Karakteristik Cluster")
        if cluster_summary.empty:
            st.warning("Tidak ada data untuk heatmap karakteristik cluster.")
        else:
            cluster_summary_clean = cluster_summary.replace([np.inf, -np.inf], np.nan).fillna(0)
            fig, ax = plt.subplots(figsize=(12, 6))
            try:
                sns.heatmap(cluster_summary_clean.T, annot=True, cmap='RdYlBu_r', fmt='.3f', cbar_kws={'label': 'Mean Value'}, ax=ax)
                ax.set_title('Karakteristik Cluster - Heatmap Mean Values', fontsize=14, fontweight='bold')
                ax.set_xlabel('Cluster', fontweight='bold')
                ax.set_ylabel('Fitur', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
            except ValueError as ve:
                st.error(f"Gagal menampilkan heatmap: {ve}")
                st.write(cluster_summary_clean)
        # Visualisasi cluster dalam ruang 2D
        st.subheader("📈 Visualisasi Cluster (PCA 2D)")
        try:
            pca_2d = PCA(n_components=2, random_state=42)
            X_pca_2d = pca_2d.fit_transform(X_processed)
            fig, ax = plt.subplots(figsize=(12, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, k_terbaik))
            for i in range(k_terbaik):
                mask = cluster_labels == i
                ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
                          c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=30)
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
        except Exception as e:
            st.error(f"Gagal melakukan visualisasi PCA 2D: {e}")
        st.subheader("💡 Insight Clustering")
        sil_score_display = max(silhouette_scores) if not all(np.isnan(silhouette_scores)) else np.nan
        st.markdown(f"""
        <div class=\"success-box\">🔍 Analisis Hasil Clustering:
        <ul>
            <li>Dataset berhasil dikelompokkan menjadi <strong>{k_terbaik} cluster</strong> yang berbeda</li>
            <li>Setiap cluster menunjukkan pola gejala dan karakteristik yang unik</li>
            <li>Silhouette Score <strong>{sil_score_display:.3f}</strong> menunjukkan kualitas pemisahan yang baik</li>
            <li>Visualisasi PCA membantu memahami distribusi cluster dalam ruang 2 dimensi</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🔬 <strong>Aplikasi Analisis Clustering Monkeypox</strong></p>
        <p>Dibuat untuk tugas akhir machine learning dengan implementasi K-Means</p>
        <p><em>Dataset: pasien dengan fitur medis</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()