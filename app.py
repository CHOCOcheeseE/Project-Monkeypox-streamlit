# ===================================================================
# CELL 1: Mengimpor Library yang Dibutuhkan
# ===================================================================

# Mari kita siapkan semua peralatan yang dibutuhkan untuk analisis kita
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # Menyembunyikan peringatan agar output lebih bersih

# Library untuk clustering dan analisis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Mengatur tampilan grafik agar lebih menarik
plt.style.use('default')
sns.set_palette("husl")

print("🔬 ANALISIS CLUSTERING MONKEYPOX")
print("=" * 50)
print("Semua library berhasil dimuat!")
print("Kita siap memulai analisis clustering untuk data monkeypox.")
print()


# ===================================================================
# CELL 2: Memuat dan Melihat Data
# ===================================================================

print("📊 LANGKAH 1: MEMUAT DATA")
print("-" * 30)

# Memuat dataset dari file CSV
df = pd.read_csv('MonkeyPox.csv')

print(f"Data berhasil dimuat!")
print(f"Jumlah pasien: {df.shape[0]}")
print(f"Jumlah kolom: {df.shape[1]}")
print()

# Mari kita lihat struktur data
print("Kolom-kolom dalam dataset:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col} ({df[col].dtype})")
print()

# Melihat distribusi kasus monkeypox
print("Distribusi kasus monkeypox:")
kasus = df['MonkeyPox'].value_counts()
for status, jumlah in kasus.items():
    persentase = (jumlah / len(df)) * 100
    print(f"  - {status}: {jumlah} pasien ({persentase:.1f}%)")
print()

# Cek apakah ada data yang hilang
data_hilang = df.isnull().sum().sum()
if data_hilang == 0:
    print("✅ Bagus! Tidak ada data yang hilang.")
else:
    print(f"⚠️  Ada {data_hilang} data yang hilang, perlu dibersihkan.")
print()

# ===================================================================
# CELL 3: Menyiapkan Data untuk Clustering
# ===================================================================

print("🛠️  LANGKAH 2: MENYIAPKAN DATA")
print("-" * 30)

# Menghapus kolom yang tidak diperlukan untuk clustering
# Kita tidak menggunakan ID pasien dan label target untuk clustering
df_fitur = df.drop(columns=["Patient_ID", "MonkeyPox"])
print(f"Fitur untuk clustering: {df_fitur.shape[1]} kolom")

# Melihat jenis data
kolom_kategori = df_fitur.select_dtypes(include=['object']).columns
kolom_numerik = df_fitur.select_dtypes(include=[np.number]).columns

print(f"Kolom kategori: {len(kolom_kategori)} kolom")
print(f"Kolom numerik: {len(kolom_numerik)} kolom")

# Mengubah data kategori menjadi angka
if len(kolom_kategori) > 0:
    print("Mengubah data kategori menjadi angka...")
    df_encoded = pd.get_dummies(df_fitur, columns=kolom_kategori, drop_first=True)
    print(f"Setelah encoding: {df_encoded.shape[1]} fitur")
else:
    df_encoded = df_fitur.copy()
    print("Tidak ada data kategori yang perlu diubah.")

# Standardisasi data agar semua fitur memiliki skala yang sama
print("Menyeragamkan skala data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)
print(f"Data telah distandarisasi dengan rata-rata: {X_scaled.mean():.6f}")
print(f"Dan standar deviasi: {X_scaled.std():.6f}")
print()


# ===================================================================
# CELL 4: Mencari Jumlah Cluster Terbaik
# ===================================================================

print("🎯 LANGKAH 3: MENCARI JUMLAH CLUSTER TERBAIK")
print("-" * 30)

# Kita akan mencoba berbagai jumlah cluster
jumlah_k = range(2, 8)  # Dari 2 sampai 7 cluster
inertia_values = []
silhouette_scores = []

print("Mencoba berbagai jumlah cluster...")
for k in jumlah_k:
    print(f"  Menguji {k} cluster...", end=" ")

    # Membuat model K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Menghitung metrik evaluasi
    inertia = kmeans.inertia_  # Mengukur kepadatan cluster
    sil_score = silhouette_score(X_scaled, labels)  # Mengukur kualitas pemisahan

    inertia_values.append(inertia)
    silhouette_scores.append(sil_score)

    print(f"Silhouette Score: {sil_score:.3f}")

# Mencari k terbaik berdasarkan silhouette score
k_terbaik = jumlah_k[np.argmax(silhouette_scores)]
score_terbaik = max(silhouette_scores)

print()
print(f"🏆 JUMLAH CLUSTER TERBAIK: {k_terbaik}")
print(f"Dengan Silhouette Score: {score_terbaik:.3f}")
print()

# ===================================================================
# CELL 5: Visualisasi Pemilihan Cluster
# ===================================================================

print("📈 LANGKAH 4: VISUALISASI PEMILIHAN CLUSTER")
print("-" * 30)

# Membuat grafik untuk membantu memahami pemilihan cluster
plt.figure(figsize=(12, 4))

# Grafik Elbow Method (untuk melihat penurunan inertia)
plt.subplot(1, 2, 1)
plt.plot(jumlah_k, inertia_values, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Jumlah Cluster')
plt.ylabel('Inertia (Kepadatan dalam Cluster)')
plt.title('Elbow Method - Mencari Jumlah Cluster Optimal')
plt.grid(True, alpha=0.3)
plt.xticks(jumlah_k)

# Grafik Silhouette Score
plt.subplot(1, 2, 2)
plt.plot(jumlah_k, silhouette_scores, 'ro-', linewidth=2, markersize=8)
plt.axvline(x=k_terbaik, color='green', linestyle='--', alpha=0.7,
           label=f'Terbaik: {k_terbaik} cluster')
plt.xlabel('Jumlah Cluster')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score - Kualitas Pemisahan Cluster')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(jumlah_k)

plt.tight_layout()
plt.show()

print("Grafik menunjukkan bahwa", k_terbaik, "cluster memberikan hasil terbaik.")
print()

# ===================================================================
# CELL 6: Melakukan Clustering Final
# ===================================================================

print("🎨 LANGKAH 5: MELAKUKAN CLUSTERING FINAL")
print("-" * 30)

# Membuat model clustering dengan jumlah cluster terbaik
kmeans_final = KMeans(n_clusters=k_terbaik, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Menambahkan hasil clustering ke dataframe
df['Cluster'] = cluster_labels

# Melihat distribusi pasien per cluster
print(f"Clustering selesai dengan {k_terbaik} cluster!")
print("Distribusi pasien per cluster:")
distribusi_cluster = pd.Series(cluster_labels).value_counts().sort_index()
for cluster_id, jumlah in distribusi_cluster.items():
    persentase = (jumlah / len(df)) * 100
    print(f"  - Cluster {cluster_id}: {jumlah} pasien ({persentase:.1f}%)")
print()

# ===================================================================
# CELL 7: Evaluasi Kualitas Clustering
# ===================================================================

print("📊 LANGKAH 6: EVALUASI KUALITAS CLUSTERING")
print("-" * 30)

# Menghitung berbagai metrik evaluasi
silhouette_final = silhouette_score(X_scaled, cluster_labels)

print("Kualitas clustering:")
print(f"  - Silhouette Score: {silhouette_final:.3f}")
print("    (Semakin mendekati 1, semakin baik pemisahan cluster)")
print()

# Membuat tabel silang antara cluster dan status monkeypox
tabel_silang = pd.crosstab(df['Cluster'], df['MonkeyPox'], margins=True)
print("Tabel Cluster vs Status Monkeypox:")
print(tabel_silang)
print()

# Menghitung persentase dalam setiap cluster
tabel_persen = pd.crosstab(df['Cluster'], df['MonkeyPox'], normalize='index') * 100
print("Persentase status monkeypox dalam setiap cluster:")
print(tabel_persen.round(1))
print()

# ===================================================================
# CELL 8: Analisis PCA dan Visualisasi Dasar
# ===================================================================

print("🔍 LANGKAH 7: ANALISIS KOMPONEN UTAMA (PCA)")
print("-" * 30)

# PCA untuk mereduksi dimensi data menjadi 2D untuk visualisasi
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Hasil PCA:")
print(f"  - Komponen 1 menjelaskan {pca.explained_variance_ratio_[0]:.1%} variasi data")
print(f"  - Komponen 2 menjelaskan {pca.explained_variance_ratio_[1]:.1%} variasi data")
print(f"  - Total variasi yang dijelaskan: {sum(pca.explained_variance_ratio_):.1%}")
print()

# Membuat visualisasi
plt.figure(figsize=(15, 5))

# Visualisasi hasil clustering
plt.subplot(1, 3, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title(f'Hasil Clustering K-Means\n({k_terbaik} Cluster)')
plt.xlabel(f'Komponen 1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'Komponen 2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.grid(True, alpha=0.3)

# Visualisasi status monkeypox asli
plt.subplot(1, 3, 2)
colors = ['red' if x == 'Positive' else 'blue' for x in df['MonkeyPox']]
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6)
plt.title('Status Monkeypox Asli')
plt.xlabel(f'Komponen 1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'Komponen 2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.legend(['Negatif', 'Positif'])
plt.grid(True, alpha=0.3)

# Visualisasi dengan centroid
plt.subplot(1, 3, 3)
centers_pca = pca.transform(kmeans_final.cluster_centers_)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200,
           linewidths=2, label='Pusat Cluster')
plt.title('Cluster dengan Pusat Cluster')
plt.xlabel(f'Komponen 1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'Komponen 2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Visualisasi PCA membantu kita melihat bagaimana cluster terbentuk dalam ruang 2D.")
print()

# ===================================================================
# CELL 9: Analisis Distribusi Detail
# ===================================================================

print("📊 LANGKAH 8: ANALISIS DISTRIBUSI DETAIL")
print("-" * 30)

plt.figure(figsize=(15, 8))

# Grafik batang distribusi
plt.subplot(2, 3, 1)
sns.countplot(data=df, x='Cluster', hue='MonkeyPox', palette='Set2')
plt.title('Distribusi Kasus Monkeypox per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Jumlah Pasien')

# Grafik persentase
plt.subplot(2, 3, 2)
prop_cluster = pd.crosstab(df['Cluster'], df['MonkeyPox'], normalize='index') * 100
prop_cluster.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='Set2')
plt.title('Persentase Status Monkeypox per Cluster')
plt.ylabel('Persentase (%)')
plt.xlabel('Cluster')
plt.legend(title='Status', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=0)

# Pie chart untuk setiap cluster
for i, cluster_id in enumerate(sorted(df['Cluster'].unique())):
    if i < 3:  # Hanya menampilkan 3 cluster pertama untuk ruang
        plt.subplot(2, 3, 4+i)
        data_cluster = df[df['Cluster'] == cluster_id]['MonkeyPox'].value_counts()
        plt.pie(data_cluster.values, labels=data_cluster.index, autopct='%1.1f%%',
               colors=['skyblue', 'lightcoral'])
        jumlah_pasien = len(df[df['Cluster'] == cluster_id])
        plt.title(f'Cluster {cluster_id}\n({jumlah_pasien} pasien)')

plt.subplot(2, 3, 3)
confusion_matrix = pd.crosstab(df['MonkeyPox'], df['Cluster'])
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matriks Kebingungan\n(Aktual vs Prediksi)')
plt.ylabel('Status Monkeypox Aktual')
plt.xlabel('Cluster Prediksi')

plt.tight_layout()
plt.show()

print("Grafik ini menunjukkan bagaimana setiap cluster berkaitan dengan status monkeypox.")
print()

# ===================================================================
# CELL 10: Analisis Fitur Penting (VERSI DIPERBAIKI)
# ===================================================================

print("🔬 LANGKAH 9: ANALISIS FITUR PENTING")
print("-" * 30)

# Menghitung rata-rata fitur untuk setiap cluster
fitur_cluster = df_encoded.copy()
fitur_cluster['Cluster'] = cluster_labels

rata_rata_cluster = fitur_cluster.groupby('Cluster').mean()
rata_rata_keseluruhan = df_encoded.mean()

print("Fitur yang paling membedakan setiap cluster:")
fitur_penting_per_cluster = {}

for cluster_id in range(k_terbaik):
    rata_cluster = rata_rata_cluster.loc[cluster_id]
    perbedaan = np.abs(rata_cluster - rata_rata_keseluruhan)
    fitur_top = perbedaan.nlargest(3)  # 3 fitur teratas
    fitur_penting_per_cluster[cluster_id] = fitur_top

    print(f"\nCluster {cluster_id}:")
    for fitur, selisih in fitur_top.items():
        nilai_cluster = rata_cluster[fitur]
        nilai_keseluruhan = rata_rata_keseluruhan[fitur]
        print(f"  - {fitur}: cluster={nilai_cluster:.3f}, rata-rata={nilai_keseluruhan:.3f}")

print()

# ===================================================================
# VISUALISASI HEATMAP YANG DIPERBAIKI
# ===================================================================

# Membuat figure dengan ukuran yang lebih besar untuk kejelasan
plt.figure(figsize=(16, 10))

# Ambil fitur-fitur penting saja untuk visualisasi yang lebih jelas
semua_fitur_penting = set()
for fitur_dict in fitur_penting_per_cluster.values():
    semua_fitur_penting.update(list(fitur_dict.keys()))

# Ambil maksimal 8 fitur untuk visualisasi yang tidak terlalu padat
fitur_terpilih = list(semua_fitur_penting)[:8]

if len(fitur_terpilih) > 0:
    # SUBPLOT 1: Heatmap Utama (diperbaiki)
    plt.subplot(2, 2, (1, 2))  # Menggunakan 2 kolom untuk heatmap utama

    # Menyiapkan data untuk heatmap
    data_heatmap = rata_rata_cluster[fitur_terpilih].T

    # Membuat heatmap dengan pengaturan yang lebih baik
    ax1 = sns.heatmap(data_heatmap,
                      annot=True,           # Menampilkan nilai numerik
                      fmt='.3f',            # Format 3 desimal untuk presisi
                      cmap='RdBu_r',        # Colormap yang kontras
                      center=0,             # Titik tengah untuk skala warna
                      cbar_kws={'label': 'Nilai Terstandar', 'shrink': 0.8},
                      linewidths=0.5,       # Garis pemisah antar sel
                      square=False,         # Bentuk persegi panjang
                      annot_kws={'size': 10, 'weight': 'bold'})  # Ukuran dan tebal teks

    # Memperbaiki label dan judul
    plt.title('Profil Karakteristik Fitur per Cluster\n(Nilai Terstandar)',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Cluster', fontsize=12, fontweight='bold')
    plt.ylabel('Fitur', fontsize=12, fontweight='bold')

    # Memutar label sumbu y agar lebih mudah dibaca
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)

    # SUBPLOT 2: Grafik Batang Perbedaan Fitur (diperbaiki)
    plt.subplot(2, 2, 3)
    if 0 in fitur_penting_per_cluster:
        fitur_0 = fitur_penting_per_cluster[0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(fitur_0)))  # Warna berbeda untuk setiap batang
        bars = plt.barh(range(len(fitur_0)), list(fitur_0.values), color=colors)

        # Menambahkan nilai pada setiap batang
        for i, (bar, value) in enumerate(zip(bars, fitur_0.values)):
            plt.text(value + 0.01, i, f'{value:.3f}',
                    va='center', fontweight='bold', fontsize=9)

        plt.yticks(range(len(fitur_0)), list(fitur_0.keys()))
        plt.xlabel('Perbedaan dari Rata-rata Keseluruhan', fontweight='bold')
        plt.title('Fitur Pembeda Cluster 0', fontweight='bold')
        plt.grid(axis='x', alpha=0.3, linestyle='--')

    # SUBPLOT 3: Distribusi Jumlah Data per Cluster
    plt.subplot(2, 2, 4)
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    colors_pie = plt.cm.Set2(np.linspace(0, 1, len(cluster_counts)))

    wedges, texts, autotexts = plt.pie(cluster_counts.values,
                                      labels=[f'Cluster {i}' for i in cluster_counts.index],
                                      autopct='%1.1f%%',
                                      colors=colors_pie,
                                      startangle=90,
                                      explode=[0.05] * len(cluster_counts))  # Sedikit memisahkan irisan

    # Memperbaiki tampilan teks pada pie chart
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    plt.title('Distribusi Data per Cluster', fontweight='bold')

    # Menyesuaikan layout agar tidak tumpang tindih
    plt.tight_layout(pad=3.0)
    plt.show()

# ===================================================================
# HEATMAP TAMBAHAN: PERBANDINGAN DENGAN RATA-RATA KESELURUHAN
# ===================================================================

# Membuat visualisasi tambahan yang menunjukkan perbedaan dengan rata-rata
plt.figure(figsize=(14, 8))

# Menghitung perbedaan setiap cluster dengan rata-rata keseluruhan
perbedaan_data = []
for cluster_id in range(k_terbaik):
    perbedaan = rata_rata_cluster.loc[cluster_id] - rata_rata_keseluruhan
    perbedaan_data.append(perbedaan[fitur_terpilih])

perbedaan_df = pd.DataFrame(perbedaan_data,
                           index=[f'Cluster {i}' for i in range(k_terbaik)],
                           columns=fitur_terpilih)

# Membuat heatmap perbedaan
ax2 = sns.heatmap(perbedaan_df.T,
                  annot=True,
                  fmt='.3f',
                  cmap='RdBu_r',
                  center=0,
                  cbar_kws={'label': 'Perbedaan dari Rata-rata Keseluruhan'},
                  linewidths=0.5,
                  annot_kws={'size': 11, 'weight': 'bold'})

plt.title('Perbedaan Karakteristik Cluster dari Rata-rata Keseluruhan\n' +
          '(Nilai Positif = Lebih Tinggi, Nilai Negatif = Lebih Rendah)',
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Cluster', fontsize=12, fontweight='bold')
plt.ylabel('Fitur', fontsize=12, fontweight='bold')
plt.yticks(rotation=0)
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

print("📊 Penjelasan Visualisasi:")
print("1. Heatmap pertama menunjukkan profil karakteristik setiap cluster")
print("2. Grafik batang menunjukkan fitur yang paling membedakan cluster tertentu")
print("3. Pie chart menunjukkan distribusi jumlah data di setiap cluster")
print("4. Heatmap kedua menunjukkan perbedaan setiap cluster dari rata-rata keseluruhan")
print("   - Warna merah: nilai lebih tinggi dari rata-rata")
print("   - Warna biru: nilai lebih rendah dari rata-rata")
print("   - Warna putih: nilai mendekati rata-rata")
print()

# ===================================================================
# CELL 11: Perbandingan dengan Hierarchical Clustering
# ===================================================================

print("🌳 LANGKAH 10: PERBANDINGAN DENGAN HIERARCHICAL CLUSTERING")
print("-" * 30)

# Menggunakan sampel kecil untuk efisiensi komputasi
ukuran_sampel = min(50, len(X_scaled))
indeks_sampel = np.random.choice(len(X_scaled), ukuran_sampel, replace=False)
data_sampel = X_scaled[indeks_sampel]
pca_sampel = X_pca[indeks_sampel]
cluster_kmeans_sampel = cluster_labels[indeks_sampel]

print(f"Menggunakan sampel {ukuran_sampel} pasien untuk perbandingan")

plt.figure(figsize=(15, 5))

# Dendrogram
plt.subplot(1, 3, 1)
linked = linkage(data_sampel, method='ward')
dendrogram(linked, truncate_mode='lastp', p=10, show_leaf_counts=True)
plt.title('Dendrogram Hierarchical Clustering')
plt.xlabel('Ukuran Cluster')
plt.ylabel('Jarak')

# Hasil K-Means pada sampel
plt.subplot(1, 3, 2)
plt.scatter(pca_sampel[:, 0], pca_sampel[:, 1], c=cluster_kmeans_sampel, cmap='viridis')
plt.title('K-Means (Sampel)')
plt.xlabel('Komponen 1')
plt.ylabel('Komponen 2')

# Hasil Hierarchical pada sampel
hierarchical_labels = fcluster(linked, k_terbaik, criterion='maxclust')
plt.subplot(1, 3, 3)
plt.scatter(pca_sampel[:, 0], pca_sampel[:, 1], c=hierarchical_labels, cmap='plasma')
plt.title('Hierarchical Clustering (Sampel)')
plt.xlabel('Komponen 1')
plt.ylabel('Komponen 2')

plt.tight_layout()
plt.show()

# Bandingkan kualitas
sil_kmeans_sampel = silhouette_score(data_sampel, cluster_kmeans_sampel)
sil_hierarchical_sampel = silhouette_score(data_sampel, hierarchical_labels)

print(f"Perbandingan kualitas clustering (pada sampel):")
print(f"  - K-Means Silhouette Score: {sil_kmeans_sampel:.3f}")
print(f"  - Hierarchical Silhouette Score: {sil_hierarchical_sampel:.3f}")

if sil_kmeans_sampel > sil_hierarchical_sampel:
    print("  - K-Means memberikan hasil yang lebih baik")
else:
    print("  - Hierarchical clustering memberikan hasil yang lebih baik")
print()

# ===================================================================
# CELL 12: Interpretasi Cluster
# ===================================================================

print("🔍 LANGKAH 11: INTERPRETASI CLUSTER")
print("-" * 30)

# Analisis setiap cluster
komposisi_cluster = pd.DataFrame({
    'Cluster': range(k_terbaik),
    'Total_Pasien': [len(df[df['Cluster'] == i]) for i in range(k_terbaik)],
    'Kasus_Positif': [len(df[(df['Cluster'] == i) & (df['MonkeyPox'] == 'Positive')]) for i in range(k_terbaik)],
    'Kasus_Negatif': [len(df[(df['Cluster'] == i) & (df['MonkeyPox'] == 'Negative')]) for i in range(k_terbaik)]
})

komposisi_cluster['Persen_Positif'] = (komposisi_cluster['Kasus_Positif'] /
                                      komposisi_cluster['Total_Pasien'] * 100).round(1)

print("Interpretasi setiap cluster:")
for idx, row in komposisi_cluster.iterrows():
    cluster_id = row['Cluster']
    total = row['Total_Pasien']
    persen_positif = row['Persen_Positif']

    print(f"\n🔸 Cluster {cluster_id}:")
    print(f"   - Total pasien: {total}")
    print(f"   - Kasus positif: {row['Kasus_Positif']} ({persen_positif}%)")
    print(f"   - Kasus negatif: {row['Kasus_Negatif']} ({100-persen_positif}%)")

    # Interpretasi risiko
    if persen_positif > 70:
        print(f"   - 🔴 RISIKO TINGGI: Mayoritas kasus positif")
    elif persen_positif < 30:
        print(f"   - 🟢 RISIKO RENDAH: Mayoritas kasus negatif")
    else:
        print(f"   - 🟡 RISIKO SEDANG: Kasus campuran")

print()

# ===================================================================
# CELL 13: Visualisasi Interpretasi
# ===================================================================

print("📊 LANGKAH 12: VISUALISASI INTERPRETASI")
print("-" * 30)

plt.figure(figsize=(12, 8))

# Grafik komposisi cluster
plt.subplot(2, 2, 1)
plt.bar(komposisi_cluster['Cluster'], komposisi_cluster['Persen_Positif'],
        color=['red' if x > 70 else 'green' if x < 30 else 'orange'
               for x in komposisi_cluster['Persen_Positif']], alpha=0.7)
plt.xlabel('Cluster')
plt.ylabel('Persentase Kasus Positif (%)')
plt.title('Profil Risiko per Cluster')
plt.grid(axis='y', alpha=0.3)

# Ukuran cluster
plt.subplot(2, 2, 2)
plt.pie(komposisi_cluster['Total_Pasien'], labels=[f'Cluster {i}' for i in range(k_terbaik)],
        autopct='%1.1f%%', startangle=90)
plt.title('Distribusi Ukuran Cluster')

# Scatter plot dengan interpretasi warna
plt.subplot(2, 2, 3)
for i in range(k_terbaik):
    mask = df['Cluster'] == i
    persen_pos = komposisi_cluster.iloc[i]['Persen_Positif']

    if persen_pos > 70:
        color = 'red'
        label = f'Cluster {i} (Risiko Tinggi)'
    elif persen_pos < 30:
        color = 'green'
        label = f'Cluster {i} (Risiko Rendah)'
    else:
        color = 'orange'
        label = f'Cluster {i} (Risiko Sedang)'

    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, alpha=0.6, label=label)

plt.xlabel('Komponen 1')
plt.ylabel('Komponen 2')
plt.title('Clustering dengan Kategori Risiko')
plt.legend(bbox_to_anchor=(1.05, 1))

# Tabel summary
plt.subplot(2, 2, 4)
plt.axis('tight')
plt.axis('off')
tabel_data = komposisi_cluster[['Cluster', 'Total_Pasien', 'Persen_Positif']].copy()
tabel_data.columns = ['Cluster', 'Jumlah Pasien', 'Positif (%)']
tabel = plt.table(cellText=tabel_data.values, colLabels=tabel_data.columns,
                 cellLoc='center', loc='center')
tabel.auto_set_font_size(False)
tabel.set_fontsize(10)
plt.title('Ringkasan Cluster')

plt.tight_layout()
plt.show()

print("Visualisasi ini membantu memahami karakteristik dan risiko setiap cluster.")
print()

# ===================================================================
# CELL 14: Kesimpulan dan Rekomendasi
# ===================================================================

print("🎯 LANGKAH 13: KESIMPULAN DAN REKOMENDASI")
print("=" * 50)

print("📋 RINGKASAN HASIL ANALISIS:")
print()

print("1. KUALITAS CLUSTERING:")
print(f"   - Jumlah cluster optimal: {k_terbaik}")
print(f"   - Silhouette Score: {silhouette_final:.3f}")
print("   - Interpretasi: ", end="")
if silhouette_final > 0.5:
    print("Kualitas clustering sangat baik")
elif silhouette_final > 0.3:
    print("Kualitas clustering cukup baik")
else:
    print("Kualitas clustering perlu diperbaiki")

print()
print("2. KARAKTERISTIK CLUSTER:")
for idx, row in komposisi_cluster.iterrows():
    cluster_id = row['Cluster']
    persen_positif = row['Persen_Positif']
    print(f"   - Cluster {cluster_id}: {row['Total_Pasien']} pasien, {persen_positif}% positif")

print()
print("3. INSIGHT KLINIS:")
cluster_risiko_tinggi = komposisi_cluster[komposisi_cluster['Persen_Positif'] > 70]
cluster_risiko_rendah = komposisi_cluster[komposisi_cluster['Persen_Positif'] < 30]

if len(cluster_risiko_tinggi) > 0:
    print(f"   - Ditemukan {len(cluster_risiko_tinggi)} cluster berisiko tinggi")
    print("   - Pasien dalam cluster ini perlu perhatian khusus")

if len(cluster_risiko_rendah) > 0:
    print(f"   - Ditemukan {len(cluster_risiko_rendah)} cluster berisiko rendah")
    print("   - Dapat digunakan sebagai grup kontrol")

print()
print("4. REKOMENDASI:")
print("   ✅ Gunakan hasil clustering untuk stratifikasi risiko pasien")
print("   ✅ Fokuskan sumber daya pada cluster berisiko tinggi")
print("   ✅ Validasi hasil dengan data klinis tambahan")
print("   ✅ Pertimbangkan faktor klinis lain dalam interpretasi")

print()
print("5. KETERBATASAN:")
print("   ⚠️  Hasil clustering bersifat eksploratif, bukan diagnostik")
print("   ⚠️  Diperlukan validasi dengan ahli medis")
print("   ⚠️  Faktor eksternal mungkin mempengaruhi hasil")