import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Konfigurasi halaman
st.set_page_config(
    page_title="🔬 MonkeyPox Clustering Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .step-header {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Dataset sample (karena tidak ada file CSV yang diberikan, saya buat sample dataset)
@st.cache_data
def load_sample_data():
    """Membuat sample dataset MonkeyPox untuk demo"""
    np.random.seed(42)
    n_samples = 500
    
    # Generate sample data
    data = {
        'Patient_ID': [f'P{i:04d}' for i in range(1, n_samples + 1)],
        'Systemic Illness': np.random.choice(['Yes', 'No', 'Unknown'], n_samples, p=[0.3, 0.6, 0.1]),
        'Rectal Pain': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'Sore Throat': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'Penile Oedema': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Oral Lesions': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'Solitary Lesion': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'Swollen Tonsils': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'HIV Infection': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'Sexually Transmitted Infection': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'MonkeyPox': np.random.choice(['Positive', 'Negative'], n_samples, p=[0.4, 0.6])
    }
    
    return pd.DataFrame(data)

# Header utama
st.markdown("""
<div class="main-header">
    <h1>🔬 ANALISIS CLUSTERING MONKEYPOX</h1>
    <p>Aplikasi Analisis Data untuk Identifikasi Pola Kasus MonkeyPox menggunakan Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x100/667eea/ffffff?text=MonkeyPox+Analysis", use_container_width=True)
    st.markdown("### 📊 Navigasi")
    
    show_data_overview = st.checkbox("📋 Overview Data", value=True)
    show_preprocessing = st.checkbox("🛠️ Preprocessing", value=True)
    show_clustering = st.checkbox("🎯 Analisis Clustering", value=True)
    show_visualization = st.checkbox("📈 Visualisasi", value=True)
    show_results = st.checkbox("📊 Hasil & Interpretasi", value=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Parameter")
    max_clusters = st.slider("Maksimal Cluster", min_value=3, max_value=10, value=7)

# Load data
df = load_sample_data()

# Main content
if show_data_overview:
    st.markdown('<div class="step-header"><h2>📊 LANGKAH 1: OVERVIEW DATA</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pasien", df.shape[0])
    with col2:
        st.metric("Jumlah Fitur", df.shape[1])
    with col3:
        positive_count = (df["MonkeyPox"] == "Positive").sum()
        st.metric("Kasus Positif", positive_count)
    with col4:
        negative_count = (df["MonkeyPox"] == "Negative").sum()
        st.metric("Kasus Negatif", negative_count)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.write("**Informasi Dataset:**")
    st.write(f"- Dataset memiliki {df.shape[0]} sampel pasien dengan {df.shape[1]} kolom")
    st.write(f"- Distribusi kasus: {positive_count} positif ({positive_count/len(df)*100:.1f}%) dan {negative_count} negatif ({negative_count/len(df)*100:.1f}%)")
    st.write("- Tidak ada data yang hilang ✅")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tampilkan sample data
    st.subheader("🔍 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Distribusi kasus
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(df, names='MonkeyPox', title='Distribusi Kasus MonkeyPox',
                        color_discrete_sequence=['#ff7f7f', '#87ceeb'])
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Heatmap korelasi fitur boolean
        boolean_cols = [col for col in df.columns if df[col].dtype in ['int64', 'bool'] and col != 'Patient_ID']
        if boolean_cols:
            corr_matrix = df[boolean_cols].corr()
            fig_corr = px.imshow(corr_matrix, 
                               title='Matriks Korelasi Fitur',
                               color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)

if show_preprocessing:
    st.markdown('<div class="step-header"><h2>🛠️ LANGKAH 2: PREPROCESSING DATA</h2></div>', unsafe_allow_html=True)
    
    # Persiapan data
    df_features = df.drop(columns=["Patient_ID", "MonkeyPox"])
    
    # Handle categorical columns
    if 'Systemic Illness' in df_features.columns:
        df_features["Systemic Illness"] = df_features["Systemic Illness"].fillna('Unknown')
        df_encoded = pd.get_dummies(df_features, columns=['Systemic Illness'])
    else:
        df_encoded = df_features.copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Fitur Sebelum Encoding:**")
        st.write(f"- Jumlah fitur: {df_features.shape[1]}")
        st.write(f"- Fitur kategori: {len(df_features.select_dtypes(include=['object']).columns)}")
        st.write(f"- Fitur numerik: {len(df_features.select_dtypes(include=[np.number]).columns)}")
    
    with col2:
        st.write("**Fitur Setelah Encoding:**")
        st.write(f"- Jumlah fitur: {df_encoded.shape[1]}")
        st.write("- Semua fitur telah dikonversi menjadi numerik")
        st.write("- Siap untuk standardisasi")
    
    # Standardisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    
    # PCA untuk reduksi dimensi
    pca = PCA(n_components=0.95)
    X_processed = pca.fit_transform(X_scaled)
    
    st.success(f"✅ Data berhasil diproses! Dimensi direduksi dari {X_scaled.shape[1]} menjadi {X_processed.shape[1]} komponen (menjelaskan {sum(pca.explained_variance_ratio_):.1%} variansi)")

if show_clustering:
    st.markdown('<div class="step-header"><h2>🎯 LANGKAH 3: ANALISIS CLUSTERING</h2></div>', unsafe_allow_html=True)
    
    # Mencari jumlah cluster optimal
    k_range = range(2, max_clusters + 1)
    inertia_values = []
    silhouette_scores = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, k in enumerate(k_range):
        status_text.text(f'Menguji {k} cluster...')
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_processed)
        
        inertia_values.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_processed, labels))
        
        progress_bar.progress((i + 1) / len(k_range))
    
    # Cluster terbaik
    best_k = k_range[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)
    
    status_text.empty()
    progress_bar.empty()
    
    st.success(f"🏆 **Jumlah Cluster Terbaik: {best_k}** dengan Silhouette Score: {best_score:.3f}")
    
    # Visualisasi pemilihan cluster
    col1, col2 = st.columns(2)
    
    with col1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=list(k_range), y=inertia_values, 
                                     mode='lines+markers', name='Inertia',
                                     line=dict(color='blue', width=3),
                                     marker=dict(size=8)))
        fig_elbow.update_layout(title='Elbow Method - Pemilihan Cluster Optimal',
                              xaxis_title='Jumlah Cluster',
                              yaxis_title='Inertia')
        st.plotly_chart(fig_elbow, use_container_width=True)
    
    with col2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(x=list(k_range), y=silhouette_scores,
                                   mode='lines+markers', name='Silhouette Score',
                                   line=dict(color='red', width=3),
                                   marker=dict(size=8)))
        fig_sil.add_vline(x=best_k, line_dash="dash", line_color="green",
                         annotation_text=f"Terbaik: {best_k}")
        fig_sil.update_layout(title='Silhouette Score - Kualitas Clustering',
                            xaxis_title='Jumlah Cluster',
                            yaxis_title='Silhouette Score')
        st.plotly_chart(fig_sil, use_container_width=True)

if show_visualization:
    st.markdown('<div class="step-header"><h2>📈 LANGKAH 4: VISUALISASI HASIL</h2></div>', unsafe_allow_html=True)
    
    # Clustering final
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_processed)
    df["Cluster"] = cluster_labels
    
    # PCA 2D untuk visualisasi
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X_scaled)
    
    # Distribusi cluster
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Distribusi pasien per cluster
        cluster_dist = pd.Series(cluster_labels).value_counts().sort_index()
        fig_bar = px.bar(x=cluster_dist.index, y=cluster_dist.values,
                        title='Distribusi Pasien per Cluster',
                        labels={'x': 'Cluster', 'y': 'Jumlah Pasien'},
                        color=cluster_dist.values, color_continuous_scale='viridis')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Visualisasi PCA dengan cluster
        df_pca = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': cluster_labels,
            'MonkeyPox': df['MonkeyPox']
        })
        
        fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster',
                           title='Visualisasi Cluster (PCA 2D)',
                           labels={'PC1': f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})',
                                 'PC2': f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})'},
                           color_continuous_scale='viridis')
        st.plotly_chart(fig_pca, use_container_width=True)
    
    with col3:
        # Status MonkeyPox vs Cluster
        crosstab = pd.crosstab(df["Cluster"], df["MonkeyPox"])
        fig_heatmap = px.imshow(crosstab.values,
                              x=crosstab.columns,
                              y=crosstab.index,
                              title='Cluster vs Status MonkeyPox',
                              color_continuous_scale='Blues',
                              text_auto=True)
        fig_heatmap.update_layout(xaxis_title='Status MonkeyPox',
                                yaxis_title='Cluster')
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Visualisasi distribusi detail
    st.subheader("📊 Analisis Detail per Cluster")
    
    fig_subplots = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Distribusi Kasus per Cluster', 'Persentase Status per Cluster',
                       'Pie Chart Cluster 0', 'Pie Chart Cluster 1'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"type": "pie"}, {"type": "pie"}]]
    )
    
    # Subplot 1: Bar chart
    for status in df['MonkeyPox'].unique():
        data = df[df['MonkeyPox'] == status]['Cluster'].value_counts().sort_index()
        fig_subplots.add_trace(
            go.Bar(x=data.index, y=data.values, name=status),
            row=1, col=1
        )
    
    # Subplot 2: Percentage bar chart
    prop_cluster = pd.crosstab(df["Cluster"], df["MonkeyPox"], normalize="index") * 100
    for status in prop_cluster.columns:
        fig_subplots.add_trace(
            go.Bar(x=prop_cluster.index, y=prop_cluster[status], name=f'{status} %'),
            row=1, col=2
        )
    
    # Subplot 3 & 4: Pie charts untuk cluster 0 dan 1
    if 0 in df['Cluster'].values:
        data_cluster_0 = df[df["Cluster"] == 0]["MonkeyPox"].value_counts()
        fig_subplots.add_trace(
            go.Pie(labels=data_cluster_0.index, values=data_cluster_0.values),
            row=2, col=1
        )
    
    if 1 in df['Cluster'].values:
        data_cluster_1 = df[df["Cluster"] == 1]["MonkeyPox"].value_counts()
        fig_subplots.add_trace(
            go.Pie(labels=data_cluster_1.index, values=data_cluster_1.values),
            row=2, col=2
        )
    
    fig_subplots.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig_subplots, use_container_width=True)

if show_results:
    st.markdown('<div class="step-header"><h2>📊 LANGKAH 5: HASIL & INTERPRETASI</h2></div>', unsafe_allow_html=True)
    
    # Evaluasi clustering
    silhouette_final = silhouette_score(X_scaled, cluster_labels)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Silhouette Score Final", f"{silhouette_final:.3f}")
        st.write("**Interpretasi:**")
        if silhouette_final > 0.7:
            st.success("Sangat baik - Cluster sangat terpisah dengan jelas")
        elif silhouette_final > 0.5:
            st.success("Baik - Cluster terpisah dengan cukup jelas")
        elif silhouette_final > 0.3:
            st.warning("Cukup - Cluster memiliki pemisahan yang moderat")
        else:
            st.error("Kurang baik - Cluster tidak terpisah dengan jelas")
    
    with col2:
        # Tabel silang
        crosstab_result = pd.crosstab(df["Cluster"], df["MonkeyPox"], margins=True)
        st.write("**Tabel Cluster vs Status MonkeyPox:**")
        st.dataframe(crosstab_result, use_container_width=True)
    
    # Analisis fitur penting
    st.subheader("🔍 Analisis Fitur Penting per Cluster")
    
    fitur_cluster = df_encoded.copy()
    fitur_cluster["Cluster"] = cluster_labels
    
    rata_rata_cluster = fitur_cluster.groupby("Cluster").mean()
    rata_rata_keseluruhan = df_encoded.mean()
    
    # Buat tabs untuk setiap cluster
    cluster_tabs = st.tabs([f"Cluster {i}" for i in range(best_k)])
    
    for i, tab in enumerate(cluster_tabs):
        with tab:
            if i in rata_rata_cluster.index:
                rata_cluster = rata_rata_cluster.loc[i]
                perbedaan = np.abs(rata_cluster - rata_rata_keseluruhan)
                fitur_top = perbedaan.nlargest(5)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write(f"**Karakteristik Cluster {i}:**")
                    jumlah_pasien = len(df[df['Cluster'] == i])
                    positif_cluster = len(df[(df['Cluster'] == i) & (df['MonkeyPox'] == 'Positive')])
                    st.write(f"- Jumlah pasien: {jumlah_pasien}")
                    st.write(f"- Kasus positif: {positif_cluster} ({positif_cluster/jumlah_pasien*100:.1f}%)")
                
                with col2:
                    # Bar chart fitur penting
                    fig_features = px.bar(
                        x=list(fitur_top.values),
                        y=list(fitur_top.index),
                        orientation='h',
                        title=f'Top 5 Fitur Pembeda Cluster {i}',
                        labels={'x': 'Perbedaan dari Rata-rata', 'y': 'Fitur'}
                    )
                    fig_features.update_layout(height=300)
                    st.plotly_chart(fig_features, use_container_width=True)
    
    # Kesimpulan
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### 📋 **KESIMPULAN ANALISIS:**")
    st.write(f"""
    1. **Jumlah Cluster Optimal:** {best_k} cluster dengan Silhouette Score {best_score:.3f}
    2. **Kualitas Clustering:** {'Sangat baik' if best_score > 0.7 else 'Baik' if best_score > 0.5 else 'Cukup' if best_score > 0.3 else 'Perlu perbaikan'}
    3. **Distribusi Data:** Dataset berhasil dikelompokkan menjadi {best_k} cluster yang berbeda
    4. **Aplikasi:** Hasil clustering dapat membantu identifikasi pola kasus MonkeyPox untuk diagnosis yang lebih tepat
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔬 MonkeyPox Clustering Analysis | Dibuat dengan ❤️ menggunakan Streamlit</p>
    <p>Untuk keperluan analisis dan penelitian medis</p>
</div>
""", unsafe_allow_html=True)