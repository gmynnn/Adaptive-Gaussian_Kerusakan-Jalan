"""
==========================================================================
  APLIKASI DETEKSI KERUSAKAN JALAN (Lubang & Retakan)
  Menggunakan: Python, Streamlit, OpenCV
  Metode: Adaptive Gaussian Thresholding + Analisis Kontur
==========================================================================
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# ========================================================================
# KONFIGURASI HALAMAN STREAMLIT
# ========================================================================
st.set_page_config(
    page_title="Deteksi Kerusakan Jalan",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
)

# ========================================================================
# CUSTOM CSS — Palet warna kalem (abu-abu & putih), UI minimalis
# ========================================================================
st.markdown("""
<style>
    /* ---- Reset dan font utama ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Header utama ---- */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: #374151;
        font-weight: 600;
        font-size: 1.6rem;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #9CA3AF;
        font-size: 0.85rem;
        margin: 0;
    }

    /* ---- Card/container untuk metrik ---- */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .metric-card .label {
        color: #9CA3AF;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        color: #1F2937;
        font-size: 1.5rem;
        font-weight: 600;
    }
    .metric-card .unit {
        color: #6B7280;
        font-size: 0.75rem;
        font-weight: 400;
    }

    /* ---- Badge keparahan ---- */
    .severity-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .severity-ringan {
        background: #F0FDF4; color: #166534; border: 1px solid #BBF7D0;
    }
    .severity-sedang {
        background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A;
    }
    .severity-berat {
        background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA;
    }
    .severity-kritis {
        background: #FDF2F8; color: #831843; border: 1px solid #FBCFE8;
    }

    /* ---- Grid matriks piksel ---- */
    .pixel-grid {
        border-collapse: collapse;
        margin: 0 auto;
        font-size: 0.68rem;
        font-family: 'Courier New', monospace;
    }
    .pixel-grid td {
        width: 38px;
        height: 28px;
        text-align: center;
        border: 1px solid #E5E7EB;
        padding: 2px 4px;
        color: #374151;
    }
    .pixel-grid .row-header, .pixel-grid .col-header {
        background: #F9FAFB;
        color: #9CA3AF;
        font-weight: 600;
        font-size: 0.6rem;
        border: 1px solid #E5E7EB;
    }
    .pixel-grid .corner {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
    }
    .pixel-diff-pos {
        background: #FEF2F2 !important;
        color: #991B1B !important;
    }
    .pixel-diff-neg {
        background: #EFF6FF !important;
        color: #1E40AF !important;
    }
    .pixel-diff-zero {
        background: #F9FAFB !important;
        color: #D1D5DB !important;
    }

    /* ---- Tabel matriks ---- */
    .matrix-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
        margin-top: 0.5rem;
    }
    .matrix-table th {
        background: #F9FAFB;
        color: #6B7280;
        font-weight: 500;
        padding: 0.6rem 0.8rem;
        text-align: left;
        border-bottom: 1px solid #E5E7EB;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .matrix-table td {
        padding: 0.55rem 0.8rem;
        border-bottom: 1px solid #F3F4F6;
        color: #374151;
    }
    .matrix-table tr:last-child td {
        border-bottom: none;
    }
    .matrix-table tr:hover td {
        background: #F9FAFB;
    }

    /* ---- Section label ---- */
    .section-label {
        color: #6B7280;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 0.6rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #E5E7EB;
    }

    /* ---- Image container ---- */
    .img-caption {
        text-align: center;
        color: #9CA3AF;
        font-size: 0.72rem;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* ---- Upload area ---- */
    [data-testid="stFileUploader"] {
        border: 2px dashed #D1D5DB;
        border-radius: 8px;
        padding: 0.5rem;
    }

    /* ---- Sembunyikan footer default ---- */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    /* ---- Divider halus ---- */
    hr {
        border: none;
        border-top: 1px solid #E5E7EB;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ========================================================================
# FUNGSI-FUNGSI PENGOLAHAN CITRA
# ========================================================================

def konversi_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """
    Mengubah citra BGR menjadi grayscale.
    Grayscale diperlukan sebagai input untuk Adaptive Thresholding.
    """
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def terapkan_adaptive_gaussian(gray: np.ndarray,
                                block_size: int = 51,
                                c_value: int = 10,
                                blur_ksize: int = 7) -> np.ndarray:
    """
    Menerapkan Adaptive Gaussian Thresholding pada citra grayscale.

    PENTING: Sebelum thresholding, citra di-blur (Gaussian Blur) terlebih
    dahulu untuk menghaluskan tekstur kerikil/aspal. Tanpa langkah ini,
    tekstur aspal akan ikut terdeteksi sebagai "kerusakan" (noise).

    Cara kerja:
    - Gaussian Blur menghaluskan detail tekstur halus (kerikil, aspal)
      sehingga hanya perbedaan intensitas besar (lubang, retakan) yang tersisa.
    - Adaptive Threshold menghitung threshold lokal berdasarkan
      rata-rata tertimbang Gaussian dari piksel tetangga
      (window block_size × block_size), dikurangi konstanta C.
    - Piksel di bawah threshold lokal → putih (255) = kerusakan
    - Piksel di atas threshold lokal → hitam (0)   = area normal

    Parameter:
    - block_size: Ukuran area tetangga (harus ganjil). Semakin besar,
                  semakin halus thresholding-nya.
    - c_value:    Konstanta yang dikurangkan dari rata-rata.
                  Semakin besar C, semakin sedikit area yang terdeteksi.
    - blur_ksize: Ukuran kernel Gaussian Blur (harus ganjil).
                  Semakin besar, semakin halus (noise hilang lebih banyak).
    """
    # Langkah pra-proses: Gaussian Blur untuk menghilangkan tekstur aspal
    # Ini KUNCI agar hasil threshold tidak penuh noise titik-titik kecil
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Adaptive Gaussian Thresholding pada citra yang sudah di-blur
    binary = cv2.adaptiveThreshold(
        blurred,                             # Citra grayscale yang sudah di-blur
        255,                                 # Nilai maksimum (putih)
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,      # Metode: Gaussian weighted
        cv2.THRESH_BINARY_INV,               # Inversi: area gelap → putih (kerusakan)
        block_size,                          # Ukuran blok tetangga
        c_value                              # Konstanta C
    )
    return binary


def bersihkan_noise_morfologi(binary: np.ndarray,
                               kernel_size: int = 3,
                               iterasi_open: int = 2,
                               iterasi_close: int = 2) -> np.ndarray:
    """
    Membersihkan noise menggunakan operasi morfologi:

    1. Opening (Erosi → Dilasi):
       - Menghapus noise kecil (kerikil, tekstur aspal halus)
       - Piksel putih kecil yang terisolasi akan dihilangkan

    2. Closing (Dilasi → Erosi):
       - Menutup lubang-lubang kecil di dalam area kerusakan
       - Menyatukan area kerusakan yang terputus-putus

    Parameter:
    - kernel_size:   Ukuran kernel (elemen struktural) untuk morfologi
    - iterasi_open:  Jumlah iterasi opening (makin banyak = makin bersih)
    - iterasi_close: Jumlah iterasi closing (makin banyak = makin menyatu)
    """
    # Buat kernel berbentuk elips (lebih natural daripada kotak)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    # Langkah 1: Opening — hapus noise kecil
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterasi_open)

    # Langkah 2: Closing — tutup celah di dalam area kerusakan
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=iterasi_close)

    return cleaned


def analisis_kontur_kerusakan(binary_clean: np.ndarray, image_bgr: np.ndarray):
    """
    Mencari kontur (garis tepi) dari area kerusakan pada citra biner,
    lalu menganalisis kontur terbesar sebagai kerusakan utama.

    Langkah:
    1. Cari semua kontur pada citra biner
    2. Filter kontur yang terlalu kecil (noise sisa)
    3. Tentukan kontur terbesar → kerusakan utama
    4. Gambar bounding box di sekitar kerusakan utama
    5. Hitung estimasi dimensi (lebar × panjang dalam piksel)

    Returns:
    - annotated:      Citra asli dengan bounding box kerusakan
    - total_damage_px: Total piksel area kerusakan (semua kontur valid)
    - largest_info:    Dict info kontur terbesar (x, y, w, h, area)
    - contour_count:   Jumlah kontur valid yang terdeteksi
    """
    # Cari semua kontur pada citra biner
    contours, _ = cv2.findContours(
        binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Buat salinan citra asli untuk anotasi (gambar bounding box)
    annotated = image_bgr.copy()

    # Hitung luas minimum kontur agar tidak menghitung noise
    # (0.01% dari total piksel citra)
    total_pixels = binary_clean.shape[0] * binary_clean.shape[1]
    min_contour_area = total_pixels * 0.0001

    # Filter kontur yang cukup besar (bukan noise)
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    total_damage_px = 0
    largest_area = 0
    largest_info = None

    for contour in valid_contours:
        area = cv2.contourArea(contour)
        total_damage_px += area

        # Hitung bounding box (persegi panjang terkecil yang mengapit kontur)
        x, y, w, h = cv2.boundingRect(contour)

        if area > largest_area:
            largest_area = area
            largest_info = {"x": x, "y": y, "w": w, "h": h, "area": area}

    # Gambar kontur kerusakan pada citra anotasi
    # Warna abu-abu gelap (70, 70, 70) agar konsisten dengan tema minimalis
    cv2.drawContours(annotated, valid_contours, -1, (70, 70, 70), 1)

    # Gambar bounding box pada kerusakan terbesar
    if largest_info:
        x, y, w, h = largest_info["x"], largest_info["y"], largest_info["w"], largest_info["h"]
        # Bounding box dengan warna abu (100, 100, 100) dan ketebalan 2 px
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (80, 80, 80), 2)
        # Label dimensi
        label = f"{w}x{h} px"
        cv2.putText(annotated, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)

    return annotated, total_damage_px, largest_info, len(valid_contours)


def hitung_metrik_perubahan(gray: np.ndarray, binary_clean: np.ndarray):
    """
    Menghitung metrik perubahan antara citra grayscale dan citra biner akhir.

    Metrik yang dihitung:
    - Rata-rata intensitas (mean)
    - Standar deviasi intensitas
    - Jumlah piksel putih/hitam
    - Rasio piksel rusak terhadap total
    """
    # ----- Statistik Citra Grayscale -----
    gray_mean = np.mean(gray)
    gray_std = np.std(gray)
    gray_median = np.median(gray)
    gray_min = int(np.min(gray))
    gray_max = int(np.max(gray))

    # ----- Statistik Citra Biner -----
    total_px = binary_clean.shape[0] * binary_clean.shape[1]
    white_px = int(np.count_nonzero(binary_clean))   # Piksel putih (kerusakan)
    black_px = total_px - white_px                    # Piksel hitam (area normal)

    bin_mean = np.mean(binary_clean)
    bin_std = np.std(binary_clean)

    rasio_rusak = (white_px / total_px) * 100  # Persentase area rusak

    return {
        "gray_mean": gray_mean,
        "gray_std": gray_std,
        "gray_median": gray_median,
        "gray_min": gray_min,
        "gray_max": gray_max,
        "bin_mean": bin_mean,
        "bin_std": bin_std,
        "total_px": total_px,
        "white_px": white_px,
        "black_px": black_px,
        "rasio_rusak": rasio_rusak,
    }


def ambil_sampel_matriks(gray: np.ndarray, binary_clean: np.ndarray,
                          ukuran: int = 8):
    """
    Mengambil sampel matriks piksel dari area tengah citra.

    Tujuan:
    - Menampilkan perbandingan nilai piksel AKTUAL antara citra grayscale
      (nilai kontinu 0–255) dan citra biner (nilai 0 atau 255).
    - Pengguna bisa melihat langsung bagaimana piksel berubah setelah
      proses Adaptive Gaussian Thresholding.

    Sampel diambil dari area TENGAH citra agar representatif
    (biasanya kerusakan berada di tengah foto).

    Parameter:
    - ukuran: jumlah baris dan kolom sampel (default 8×8)

    Returns:
    - gray_sample:   array ukuran × ukuran dari citra grayscale
    - binary_sample: array ukuran × ukuran dari citra biner
    - start_row, start_col: posisi awal sampel pada citra asli
    """
    h, w = gray.shape
    # Ambil dari titik tengah citra
    start_row = max(0, (h // 2) - (ukuran // 2))
    start_col = max(0, (w // 2) - (ukuran // 2))
    end_row = min(h, start_row + ukuran)
    end_col = min(w, start_col + ukuran)

    gray_sample = gray[start_row:end_row, start_col:end_col]
    binary_sample = binary_clean[start_row:end_row, start_col:end_col]

    return gray_sample, binary_sample, start_row, start_col


def render_pixel_grid_html(matrix: np.ndarray, start_row: int, start_col: int,
                            title: str, css_class: str = "") -> str:
    """
    Membuat tabel HTML berisi nilai piksel dari sebuah matriks.
    Setiap sel menampilkan nilai piksel aktual (0–255).
    Baris dan kolom header menunjukkan indeks koordinat piksel di citra asli.
    """
    rows, cols = matrix.shape
    html = f'<table class="pixel-grid {css_class}">'

    # Header kolom (indeks kolom piksel)
    html += '<tr><td class="corner"></td>'
    for c in range(cols):
        html += f'<td class="col-header">{start_col + c}</td>'
    html += '</tr>'

    # Baris data
    for r in range(rows):
        html += f'<tr><td class="row-header">{start_row + r}</td>'
        for c in range(cols):
            val = int(matrix[r, c])
            # Warna latar belakang proporsional dengan intensitas
            bg_intensity = val  # 0=hitam, 255=putih
            # Teks putih jika gelap, hitam jika terang
            text_color = "#F9FAFB" if bg_intensity < 128 else "#1F2937"
            html += (f'<td style="background:rgb({bg_intensity},{bg_intensity},{bg_intensity});'
                     f'color:{text_color};">{val}</td>')
        html += '</tr>'
    html += '</table>'
    return html


def render_diff_grid_html(gray_sample: np.ndarray, binary_sample: np.ndarray,
                           start_row: int, start_col: int) -> str:
    """
    Membuat tabel HTML matriks selisih (perbedaan) antara piksel
    citra biner dan citra grayscale.

    Warna sel:
    - Merah muda: selisih positif (piksel jadi lebih terang → terdeteksi kerusakan)
    - Biru muda:  selisih negatif (piksel jadi lebih gelap)
    - Abu-abu:    tidak berubah (selisih = 0)
    """
    diff = binary_sample.astype(np.int16) - gray_sample.astype(np.int16)
    rows, cols = diff.shape
    html = '<table class="pixel-grid">'

    # Header kolom
    html += '<tr><td class="corner"></td>'
    for c in range(cols):
        html += f'<td class="col-header">{start_col + c}</td>'
    html += '</tr>'

    # Baris data
    for r in range(rows):
        html += f'<tr><td class="row-header">{start_row + r}</td>'
        for c in range(cols):
            val = int(diff[r, c])
            if val > 0:
                css = "pixel-diff-pos"
            elif val < 0:
                css = "pixel-diff-neg"
            else:
                css = "pixel-diff-zero"
            html += f'<td class="{css}">{val:+d}</td>'
        html += '</tr>'
    html += '</table>'
    return html


def tentukan_keparahan(rasio_rusak: float) -> tuple:
    """
    Menentukan tingkat keparahan kerusakan jalan berdasarkan
    persentase piksel area yang terdeteksi rusak.

    Klasifikasi:
    - Ringan  : < 5%   — kerusakan minor, retakan kecil
    - Sedang  : 5–15%  — kerusakan moderat, lubang kecil/retakan melebar
    - Berat   : 15–30% — kerusakan signifikan, lubang besar
    - Kritis  : > 30%  — kerusakan parah, perlu perbaikan segera

    Returns:
    - (label, css_class) untuk ditampilkan di UI
    """
    if rasio_rusak < 5:
        return "Ringan", "severity-ringan"
    elif rasio_rusak < 15:
        return "Sedang", "severity-sedang"
    elif rasio_rusak < 30:
        return "Berat", "severity-berat"
    else:
        return "Kritis", "severity-kritis"


# ========================================================================
# ANTARMUKA UTAMA (USER INTERFACE)
# ========================================================================

def main():
    # ---- Header Aplikasi ----
    st.markdown("""
    <div class="main-header">
        <h1>🛣️ Deteksi Kerusakan Jalan</h1>
        <p>Analisis lubang dan retakan menggunakan Adaptive Gaussian Thresholding</p>
    </div>
    """, unsafe_allow_html=True)

    # ================================================================
    # BAGIAN 1: UPLOAD GAMBAR
    # ================================================================
    col_upload_l, col_upload_m, col_upload_r = st.columns([1, 2, 1])
    with col_upload_m:
        uploaded_file = st.file_uploader(
            "Unggah citra jalan",
            type=["jpg", "jpeg", "png"],
            help="Format yang didukung: JPG, JPEG, PNG",
            label_visibility="collapsed",
        )

    if uploaded_file is None:
        # Tampilan placeholder saat belum ada gambar
        st.markdown("---")
        col_ph1, col_ph2, col_ph3 = st.columns([1, 2, 1])
        with col_ph2:
            st.markdown(
                """
                <div style="text-align:center; padding: 3rem 1rem; color: #9CA3AF;">
                    <p style="font-size: 2.5rem; margin-bottom: 0.5rem;">📷</p>
                    <p style="font-size: 0.85rem;">
                        Unggah gambar citra jalan rusak untuk memulai analisis.<br>
                        <span style="font-size: 0.75rem; color: #D1D5DB;">
                            Mendukung format .jpg, .jpeg, .png
                        </span>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    # ================================================================
    # BAGIAN 2: BACA & PROSES GAMBAR
    # ================================================================

    # Baca file yang diunggah menjadi array NumPy (format BGR untuk OpenCV)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("⚠️ Gagal membaca gambar. Pastikan file tidak rusak.")
        return

    # ---- Sidebar: Parameter Pengolahan ----
    with st.sidebar:
        st.markdown("#### Parameter Pengolahan")
        st.markdown("---")

        st.markdown(
            '<p style="color:#6B7280; font-size:0.72rem; text-transform:uppercase; '
            'letter-spacing:0.04em; font-weight:600;">Gaussian Blur (Pra-proses)</p>',
            unsafe_allow_html=True,
        )
        blur_ksize = st.slider(
            "Blur Kernel Size",
            min_value=1, max_value=21, value=7, step=2,
            help="Ukuran kernel Gaussian Blur. Semakin besar → tekstur aspal semakin hilang.",
        )

        st.markdown(
            '<p style="color:#6B7280; font-size:0.72rem; text-transform:uppercase; '
            'letter-spacing:0.04em; font-weight:600; margin-top:1rem;">Adaptive Threshold</p>',
            unsafe_allow_html=True,
        )
        block_size = st.slider(
            "Block Size",
            min_value=3, max_value=99, value=51, step=2,
            help="Ukuran area tetangga (harus ganjil). Semakin besar → semakin halus.",
        )
        c_value = st.slider(
            "Konstanta C",
            min_value=1, max_value=30, value=10,
            help="Nilai yang dikurangkan dari rata-rata lokal. Semakin besar → semakin sedikit area terdeteksi.",
        )

        st.markdown(
            '<p style="color:#6B7280; font-size:0.72rem; text-transform:uppercase; '
            'letter-spacing:0.04em; font-weight:600; margin-top:1rem;">Morfologi</p>',
            unsafe_allow_html=True,
        )
        kernel_size = st.slider(
            "Kernel Size",
            min_value=1, max_value=15, value=5, step=2,
            help="Ukuran elemen struktural untuk operasi morfologi.",
        )
        iterasi_open = st.slider(
            "Iterasi Opening",
            min_value=0, max_value=5, value=2,
            help="Jumlah iterasi Opening (hapus noise kecil).",
        )
        iterasi_close = st.slider(
            "Iterasi Closing",
            min_value=0, max_value=5, value=2,
            help="Jumlah iterasi Closing (tutup celah).",
        )

    # ---- Langkah 1: Konversi ke Grayscale ----
    gray = konversi_grayscale(image_bgr)

    # ---- Langkah 2: Adaptive Gaussian Thresholding (dengan Blur pra-proses) ----
    binary = terapkan_adaptive_gaussian(gray, block_size, c_value, blur_ksize)

    # ---- Langkah 3: Operasi Morfologi (pembersihan noise) ----
    binary_clean = bersihkan_noise_morfologi(
        binary, kernel_size, iterasi_open, iterasi_close
    )

    # ---- Langkah 4: Analisis kontur kerusakan ----
    annotated, total_damage_px, largest_info, contour_count = \
        analisis_kontur_kerusakan(binary_clean, image_bgr)

    # ---- Langkah 5: Hitung metrik perubahan ----
    metrik = hitung_metrik_perubahan(gray, binary_clean)
    severity_label, severity_class = tentukan_keparahan(metrik["rasio_rusak"])

    # ================================================================
    # BAGIAN 3: VISUALISASI — Perbandingan Gambar (Side-by-Side)
    # ================================================================
    st.markdown("---")
    st.markdown('<p class="section-label">Perbandingan Citra</p>', unsafe_allow_html=True)

    col_img1, col_img2 = st.columns(2, gap="medium")

    with col_img1:
        # Konversi BGR → RGB untuk ditampilkan dengan benar di Streamlit
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, use_container_width=True)
        st.markdown('<p class="img-caption">Citra Asli</p>', unsafe_allow_html=True)

    with col_img2:
        # Tampilkan hasil Adaptive Gaussian + Morfologi
        st.image(binary_clean, use_container_width=True, clamp=True)
        st.markdown(
            '<p class="img-caption">Hasil Adaptive Gaussian Thresholding</p>',
            unsafe_allow_html=True,
        )

    # ================================================================
    # BAGIAN 4: CITRA ANOTASI (Bounding Box Kerusakan)
    # ================================================================
    if largest_info:
        st.markdown("---")
        st.markdown(
            '<p class="section-label">Deteksi Kontur &amp; Bounding Box</p>',
            unsafe_allow_html=True,
        )
        col_ann_l, col_ann_m, col_ann_r = st.columns([1, 2, 1])
        with col_ann_m:
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True)
            st.markdown(
                '<p class="img-caption">Kontur kerusakan dengan bounding box pada lubang terbesar</p>',
                unsafe_allow_html=True,
            )

    # ================================================================
    # BAGIAN 5: RINGKASAN METRIK (Kartu Metrik)
    # ================================================================
    st.markdown("---")
    st.markdown('<p class="section-label">Ringkasan Analisis</p>', unsafe_allow_html=True)

    # ---- Baris metrik utama ----
    m1, m2, m3, m4 = st.columns(4, gap="medium")

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Area Kerusakan</div>
            <div class="value">{metrik['rasio_rusak']:.2f}<span class="unit"> %</span></div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Tingkat Keparahan</div>
            <div class="value">
                <span class="severity-badge {severity_class}">{severity_label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        if largest_info:
            dim_text = f"{largest_info['w']} × {largest_info['h']}"
        else:
            dim_text = "—"
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Dimensi Lubang Terbesar</div>
            <div class="value">{dim_text}<span class="unit"> px</span></div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Jumlah Kerusakan</div>
            <div class="value">{contour_count}<span class="unit"> area</span></div>
        </div>
        """, unsafe_allow_html=True)

    # ================================================================
    # BAGIAN 6: STATISTIK PERUBAHAN (Ringkasan Numerik)
    # ================================================================
    st.markdown("---")
    st.markdown(
        '<p class="section-label">Statistik Perubahan Piksel</p>',
        unsafe_allow_html=True,
    )

    st.markdown(f"""
    <table class="matrix-table">
        <thead>
            <tr>
                <th>Metrik</th>
                <th>Citra Grayscale</th>
                <th>Citra Biner (Threshold)</th>
                <th>Perubahan</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Rata-rata Intensitas</td>
                <td>{metrik['gray_mean']:.2f}</td>
                <td>{metrik['bin_mean']:.2f}</td>
                <td>{metrik['bin_mean'] - metrik['gray_mean']:+.2f}</td>
            </tr>
            <tr>
                <td>Standar Deviasi</td>
                <td>{metrik['gray_std']:.2f}</td>
                <td>{metrik['bin_std']:.2f}</td>
                <td>{metrik['bin_std'] - metrik['gray_std']:+.2f}</td>
            </tr>
            <tr>
                <td>Piksel Putih (Kerusakan)</td>
                <td>—</td>
                <td>{metrik['white_px']:,}</td>
                <td>—</td>
            </tr>
            <tr>
                <td>Piksel Hitam (Normal)</td>
                <td>—</td>
                <td>{metrik['black_px']:,}</td>
                <td>—</td>
            </tr>
            <tr>
                <td><strong>Rasio Area Rusak</strong></td>
                <td>—</td>
                <td><strong>{metrik['rasio_rusak']:.2f}%</strong></td>
                <td>—</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    # ================================================================
    # BAGIAN 6B: MATRIKS PIKSEL (Sampel Nilai Piksel Aktual)
    # ================================================================
    st.markdown("---")
    st.markdown(
        '<p class="section-label">Matriks Piksel — Perbandingan Nilai Piksel (Sampel 8×8 dari Tengah Citra)</p>',
        unsafe_allow_html=True,
    )

    # Ambil sampel matriks 8×8 dari area tengah citra
    gray_sample, bin_sample, sr, sc = ambil_sampel_matriks(gray, binary_clean, ukuran=8)

    st.markdown(
        f'<p style="color:#9CA3AF; font-size:0.72rem; margin-bottom:1rem;">'
        f'Sampel diambil dari piksel baris {sr}–{sr+7}, kolom {sc}–{sc+7} '
        f'(area tengah citra {gray.shape[1]}×{gray.shape[0]} px)</p>',
        unsafe_allow_html=True,
    )

    col_m1, col_m2, col_m3 = st.columns(3, gap="medium")

    with col_m1:
        st.markdown(
            '<p style="text-align:center; color:#6B7280; font-size:0.72rem; '
            'font-weight:600; text-transform:uppercase; letter-spacing:0.04em; '
            'margin-bottom:0.5rem;">Grayscale (Asli)</p>',
            unsafe_allow_html=True,
        )
        html_gray = render_pixel_grid_html(gray_sample, sr, sc, "Grayscale")
        st.markdown(html_gray, unsafe_allow_html=True)

    with col_m2:
        st.markdown(
            '<p style="text-align:center; color:#6B7280; font-size:0.72rem; '
            'font-weight:600; text-transform:uppercase; letter-spacing:0.04em; '
            'margin-bottom:0.5rem;">Biner (Adaptive Gaussian)</p>',
            unsafe_allow_html=True,
        )
        html_bin = render_pixel_grid_html(bin_sample, sr, sc, "Biner")
        st.markdown(html_bin, unsafe_allow_html=True)

    with col_m3:
        st.markdown(
            '<p style="text-align:center; color:#6B7280; font-size:0.72rem; '
            'font-weight:600; text-transform:uppercase; letter-spacing:0.04em; '
            'margin-bottom:0.5rem;">Selisih (Biner − Grayscale)</p>',
            unsafe_allow_html=True,
        )
        html_diff = render_diff_grid_html(gray_sample, bin_sample, sr, sc)
        st.markdown(html_diff, unsafe_allow_html=True)

    st.markdown(
        '<p style="color:#D1D5DB; font-size:0.65rem; margin-top:0.8rem;">'
        '<span style="color:#991B1B;">■</span> Merah = piksel naik (terdeteksi kerusakan) &nbsp;&nbsp;'
        '<span style="color:#1E40AF;">■</span> Biru = piksel turun &nbsp;&nbsp;'
        '<span style="color:#D1D5DB;">■</span> Abu = tidak berubah</p>',
        unsafe_allow_html=True,
    )

    # ================================================================
    # BAGIAN 7: DETAIL KONTUR TERBESAR
    # ================================================================
    if largest_info:
        st.markdown("---")
        st.markdown(
            '<p class="section-label">Detail Kerusakan Terbesar</p>',
            unsafe_allow_html=True,
        )

        d1, d2, d3, d4 = st.columns(4, gap="medium")

        with d1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Posisi (X, Y)</div>
                <div class="value" style="font-size:1.1rem;">
                    ({largest_info['x']}, {largest_info['y']})
                </div>
            </div>
            """, unsafe_allow_html=True)

        with d2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Lebar</div>
                <div class="value">{largest_info['w']}<span class="unit"> px</span></div>
            </div>
            """, unsafe_allow_html=True)

        with d3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Panjang</div>
                <div class="value">{largest_info['h']}<span class="unit"> px</span></div>
            </div>
            """, unsafe_allow_html=True)

        with d4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Luas Kontur</div>
                <div class="value">{largest_info['area']:,.0f}<span class="unit"> px²</span></div>
            </div>
            """, unsafe_allow_html=True)

    # ---- Footer ----
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; color:#D1D5DB; font-size:0.7rem; padding:0.5rem 0;">'
        'Adaptive Gaussian Thresholding · Computer Vision & NLP · 2026'
        '</p>',
        unsafe_allow_html=True,
    )


# ========================================================================
# ENTRY POINT
# ========================================================================
if __name__ == "__main__":
    main()
