import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import io
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ===================================
# KONFIGURASI HALAMAN
# ===================================
st.set_page_config(
    page_title="🐒 MonkeyPox Clustering Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling yang menarik
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stAlert {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .section-header {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .download-section {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
</style>
""", unsafe_allow_html=True)

# ===================================
# FUNGSI UTILITAS
# ===================================
@st.cache_data
def load_sample_data():
    """Generate sample MonkeyPox dataset untuk demo"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample data
    data = {
        'Patient_ID': [f'P{i:04d}' for i in range(1, n_samples + 1)],
        'Systemic Illness': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'Rectal Pain': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'Sore Throat': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'Penile Oedema': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Oral Lesions': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'Solitary Lesion': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'Swollen Tonsils': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'HIV Infection': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'Sexually Transmitted Infection': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
    }
    
    # Generate target variable based on features (untuk membuat pola yang realistis)
    risk_score = (data['Systemic Illness'] * 0.3 + 
                  data['Oral Lesions'] * 0.4 + 
                  data['HIV Infection'] * 0.5 + 
                  data['Sexually Transmitted Infection'] * 0.3 +
                  np.random.normal(0, 0.2, n_samples))
    
    data['MonkeyPox'] = ['Positive' if score > 0.5 else 'Negative' for score in risk_score]
    
    return pd.DataFrame(data)

def create_download_link(df, filename):
    """Membuat link download untuk dataset"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv" style="text-decoration: none; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.5rem 1rem; border-radius: 5px; font-weight: bold;">📥 Download {filename}</a>'
    return href

def plot_clustering_metrics(k_range, inertia_values, silhouette_scores, k_best):
    """Membuat plot interaktif untuk metrik clustering"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Elbow Method - Inertia', 'Silhouette Score Analysis'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Elbow plot
    fig.add_trace(
        go.Scatter(x=list(k_range), y=inertia_values, mode='lines+markers',
                  name='Inertia', line=dict(color='#4facfe', width=3),
                  marker=dict(size=8)),
        row=1, col=1
    )
    
    # Silhouette plot
    fig.add_trace(
        go.Scatter(x=list(k_range), y=silhouette_scores, mode='lines+markers',
                  name='Silhouette Score', line=dict(color='#f093fb', width=3),
                  marker=dict(size=8)),
        row=1, col=2
    )
    
    # Highlight best k
    fig.add_vline(x=k_best, line_dash="dash", line_color="green", 
                  annotation_text=f"Best K={k_best}", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False, 
                     title_text="Clustering Optimization Metrics")
    fig.update_xaxes(title_text="Number of Clusters")
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    
    return fig

def plot_pca_results(X_pca, cluster_labels, df, pca_2d):
    """Membuat visualisasi PCA interaktif"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Clustering Results in PCA Space', 'Original MonkeyPox Status'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot clustering results
    fig.add_trace(
        go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1], mode='markers',
                  marker=dict(color=cluster_labels, colorscale='viridis', size=6),
                  name='Clusters', text=[f'Cluster {c}' for c in cluster_labels]),
        row=1, col=1
    )
    
    # Plot original status
    colors = ['red' if status == 'Positive' else 'blue' for status in df['MonkeyPox']]
    fig.add_trace(
        go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1], mode='markers',
                  marker=dict(color=colors, size=6),
                  name='MonkeyPox Status', text=df['MonkeyPox']),
        row=1, col=2
    )
    
    explained_var_1 = pca_2d.explained_variance_ratio_[0]
    explained_var_2 = pca_2d.explained_variance_ratio_[1]
    
    fig.update_xaxes(title_text=f"PC1 ({explained_var_1:.1%})", row=1, col=1)
    fig.update_yaxes(title_text=f"PC2 ({explained_var_2:.1%})", row=1, col=1)
    fig.update_xaxes(title_text=f"PC1 ({explained_var_1:.1%})", row=1, col=2)
    fig.update_yaxes(title_text=f"PC2 ({explained_var_2:.1%})", row=1, col=2)
    
    fig.update_layout(height=500, showlegend=False)
    
    return fig

# ===================================
# HEADER APLIKASI
# ===================================
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">🐒 MonkeyPox Clustering Analysis</h1>
    <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem;">Advanced Machine Learning Analysis for Disease Pattern Recognition</p>
</div>
""", unsafe_allow_html=True)

# ===================================
# SIDEBAR
# ===================================
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
        <h3 style="color: white; margin: 0;">🛠️ Controls</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset section
    st.markdown("### 📊 Dataset Options")
    use_sample = st.radio(
        "Choose data source:",
        ["Use Sample Dataset", "Upload Your Own CSV"],
        help="Sample dataset is provided for demonstration purposes"
    )
    
    if use_sample == "Upload Your Own CSV":
        uploaded_file = st.file_uploader(
            "Upload MonkeyPox CSV file",
            type=['csv'],
            help="Make sure your CSV has the required columns"
        )
    else:
        uploaded_file = None
    
    st.markdown("---")
    
    # Analysis parameters
    min_clusters = st.slider("Minimum clusters", 2, 5, 2)
    max_clusters = st.slider("Maximum clusters", 6, 10, 8)
    pca_components = st.slider("PCA components (variance %)", 80, 99, 95)
    
    st.markdown("---")
    
    # About section
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        **Features:**
        - 🔍 K-Means Clustering Analysis
        - 📈 PCA Dimensionality Reduction
        - 📊 Interactive Visualizations
        - 📥 Dataset Download Options
        - 🎯 Clustering Optimization
        
        **Developer:** AI Assistant
        """, unsafe_allow_html=True)

# ===================================
# DATA LOADING DAN PREPROCESSING
# ===================================
st.markdown("""
<div class="section-header">
    📥 Dataset Download Section
</div>
""", unsafe_allow_html=True)

# Load or sample data
if 'df' not in st.session_state:
    # Load once
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Custom dataset loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            df = load_sample_data()
            st.info("🔄 Using sample dataset instead.")
    else:
        df = load_sample_data()
        st.info("📊 Using sample dataset for demonstration.")
    st.session_state.df = df
else:
    df = st.session_state.df

# Dataset download & preview
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="download-section">
            <h3 style="text-align: center; margin-top: 0; color: #2d5016;">🎯 Get Your Dataset Ready!</h3>
            <p style="text-align: center; color: #2d5016;">
                Download the sample MonkeyPox dataset to start your analysis immediately, 
                or use your own CSV file with similar structure.
            </p>
        </div>
        """, unsafe_allow_html=True)
        sample_df = load_sample_data()
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(create_download_link(sample_df, "MonkeyPox_Sample_Dataset"), unsafe_allow_html=True)
        with col_b:
            if st.button("🔍 Preview Sample Data", type="secondary"):
                st.dataframe(sample_df.head(), use_container_width=True)

# Overview metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>👥 Total Patients</h3>
        <h2>{df.shape[0]}</h2>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>📊 Features</h3>
        <h2>{df.shape[1] - 2}</h2>
    </div>
    """, unsafe_allow_html=True)
with col3:
    positive_cases = (df['MonkeyPox'] == 'Positive').sum()
    st.markdown(f"""
    <div class="metric-card">
        <h3>🔴 Positive Cases</h3>
        <h2>{positive_cases}</h2>
    </div>
    """, unsafe_allow_html=True)
with col4:
    positive_rate = (positive_cases / len(df)) * 100
    st.markdown(f"""
    <div class="metric-card">
        <h3>📈 Positive Rate</h3>
        <h2>{positive_rate:.1f}%</h2>
    </div>
    """, unsafe_allow_html=True)

# ================================
# PREPROCESSING untuk clustering
# ================================
df_features = df.drop(columns=["Patient_ID", "MonkeyPox"])
categorical_cols = df_features.select_dtypes(include=["object"]).columns
if len(categorical_cols) > 0:
    df_encoded = pd.get_dummies(df_features, columns=categorical_cols)
else:
    df_encoded = df_features.copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# PCA untuk preprocessing
pca = PCA(n_components=pca_components/100)
X_processed = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_.sum()

# Cari optimal k
k_range = range(min_clusters, max_clusters + 1)
inertia_values = []
silhouette_scores = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_processed)
    inertia_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_processed, labels))
# Tentukan best k
k_best = k_range[np.argmax(silhouette_scores)]
best_score = max(silhouette_scores)

# Final clustering dengan k_best
kmeans_final = KMeans(n_clusters=k_best, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_processed)

# PCA 2D untuk visualisasi
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)

# Prepare df_viz
df_viz = df.copy()
df_viz['Cluster'] = cluster_labels

# ================================
# TABS UNTUK ANALISIS
# ================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Overview", 
    "🎯 Clustering Analysis", 
    "📊 PCA Visualization", 
    "📈 Results Analysis",
    "🔍 Feature Importance"
])

with tab1:
    st.markdown("### 📊 Dataset Overview")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        st.markdown("### 📈 MonkeyPox Distribution")
        status_counts = df['MonkeyPox'].value_counts()
        fig = px.pie(values=status_counts.values, names=status_counts.index,
                     color_discrete_sequence=['#ff7f7f', '#87ceeb'],
                     title="Distribution of MonkeyPox Cases")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 📊 Data Statistics")
    st.dataframe(df.describe(), use_container_width=True)

with tab2:
    st.markdown("### 🎯 K-Means Clustering Analysis")
    st.info(f"🔄 Data processed: {X_processed.shape[1]} components explaining {explained_variance:.1%} of variance")
    st.success(f"🏆 Optimal number of clusters: **{k_best}** (Silhouette Score: {best_score:.3f})")
    fig_metrics = plot_clustering_metrics(k_range, inertia_values, silhouette_scores, k_best)
    st.plotly_chart(fig_metrics, use_container_width=True)

with tab3:
    st.markdown("### 📊 PCA Visualization")
    fig_pca = plot_pca_results(X_pca, cluster_labels, df_viz, pca_2d)
    st.plotly_chart(fig_pca, use_container_width=True)
    st.markdown("### 🎯 Cluster Distribution")
    col1, col2 = st.columns(2)
    with col1:
        cluster_dist = pd.Series(cluster_labels).value_counts().sort_index()
        fig_cluster = px.bar(x=cluster_dist.index, y=cluster_dist.values,
                             labels={'x': 'Cluster', 'y': 'Number of Patients'},
                             title="Patients per Cluster",
                             color=cluster_dist.values,
                             color_continuous_scale='viridis')
        st.plotly_chart(fig_cluster, use_container_width=True)
    with col2:
        crosstab = pd.crosstab(df_viz['Cluster'], df_viz['MonkeyPox'])
        fig_cross = px.bar(crosstab, barmode='group',
                          title="MonkeyPox Status by Cluster",
                          labels={'value': 'Count', 'index': 'Cluster'})
        st.plotly_chart(fig_cross, use_container_width=True)

with tab4:
    st.markdown("### 📈 Clustering Results Analysis")
    final_silhouette = silhouette_score(X_processed, cluster_labels)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Silhouette Score", f"{final_silhouette:.3f}")
    with col2:
        st.metric("🔢 Number of Clusters", k_best)
    with col3:
        st.metric("📊 Features Used", X_processed.shape[1])
    st.markdown("### 🔍 Detailed Cluster Analysis")
    crosstab_detailed = pd.crosstab(df_viz['Cluster'], df_viz['MonkeyPox'], margins=True)
    st.dataframe(crosstab_detailed, use_container_width=True)
    st.markdown("### 📊 Percentage Distribution")
    crosstab_pct = pd.crosstab(df_viz['Cluster'], df_viz['MonkeyPox'], normalize='index') * 100
    st.dataframe(crosstab_pct.round(1), use_container_width=True)

with tab5:
    st.markdown("### 🔍 Feature Importance Analysis")
    feature_cluster = df_encoded.copy()
    feature_cluster['Cluster'] = cluster_labels
    cluster_means = feature_cluster.groupby('Cluster').mean()
    overall_mean = df_encoded.mean()
    st.markdown("### 📋 Most Distinguishing Features per Cluster")
    for cluster_id in range(k_best):
        with st.expander(f"🎯 Cluster {cluster_id} Analysis"):
            cluster_mean = cluster_means.loc[cluster_id]
            differences = np.abs(cluster_mean - overall_mean)
            top_features = differences.nlargest(5)
            comparison_data = []
            for feature in top_features.index:
                comparison_data.append({
                    'Feature': feature,
                    'Cluster Mean': f"{cluster_mean[feature]:.3f}",
                    'Overall Mean': f"{overall_mean[feature]:.3f}",
                    'Difference': f"{differences[feature]:.3f}"
                })
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            fig_importance = px.bar(
                x=top_features.values,
                y=top_features.index,
                orientation='h',
                title=f"Top Features untuk Cluster {cluster_id}",
                labels={'x': 'Difference from Overall Mean', 'y': 'Features'}
            )
            # … akhir of tab5 loop …
            fig_importance.update_layout(height=300)
            st.plotly_chart(fig_importance, use_container_width=True)

# ===================================
# DOWNLOAD RESULTS
# ===================================
st.markdown("""
<div class="section-header">
    📥 Download Analysis Results
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    results_df = df.copy()
    results_df['Cluster'] = cluster_labels
    st.markdown(create_download_link(results_df, "MonkeyPox_Clustering_Results"), unsafe_allow_html=True)
with col2:
    summary_df = pd.crosstab(results_df['Cluster'], results_df['MonkeyPox'])
    summary_df_reset = summary_df.reset_index()
    st.markdown(create_download_link(summary_df_reset, "Cluster_Summary"), unsafe_allow_html=True)
with col3:
    importance_data = []
    for cluster_id in range(k_best):
        cluster_mean = cluster_means.loc[cluster_id]
        differences = np.abs(cluster_mean - overall_mean)
        top3 = differences.nlargest(3)
        for feature, diff in top3.items():
            importance_data.append({
                'Cluster': cluster_id,
                'Feature': feature,
                'Importance': diff
            })
    importance_df = pd.DataFrame(importance_data)
    st.markdown(create_download_link(importance_df, "Feature_Importance"), unsafe_allow_html=True)

# ===================================
# FOOTER
# ===================================
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🧬 MonkeyPox Clustering Analysis | Powered by Streamlit & Machine Learning</p>
    <p>Built with ❤️ for Healthcare Data Analysis</p>
</div>
""", unsafe_allow_html=True)
