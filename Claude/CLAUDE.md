# CLAUDE.md — Konteks Proyek Skripsi Bridge Bidding

## Identitas Proyek
- **Judul**: Prediksi Kontrak Terbaik pada Permainan Bridge Menggunakan Algoritma Random Forest Berbasis Data BBO
- **Jenjang**: Skripsi S1 — Ilmu Komputer
- **Bahasa penulisan**: Bahasa Indonesia
- **Timeline**: 3-4 bulan
- **Penulis**: Mahasiswa yang aktif berkompetisi bridge (paham domain secara mendalam)
- **Level teknis**: Familiar dasar Python (pernah pakai untuk tugas kuliah), belum pernah pakai ML library secara langsung

---

## Deskripsi Singkat
Penelitian ini bertujuan memprediksi **kontrak terbaik (best contract)** dalam permainan bridge menggunakan algoritma **2-stage Random Forest**. Data mentah berupa file `.lin` dari Bridge Base Online (BBO) yang berisi rekaman lengkap permainan bridge (distribusi kartu, bidding sequence, vulnerability, dealer, card play). Model terdiri dari dua tahap: (1) prediksi suit kontrak, (2) prediksi level/kategori kontrak.

---

## Sumber Data
- **Format**: File `.lin` (format BBO / Bridge Base Online)
- **Sumber**: Langsung dari BBO (turnamen online dan team match)
- **Lokasi file**: Semua file `.lin` ada di direktori proyek
- **Jumlah**: Saat ini ~25 file, tapi penulis memiliki lebih banyak data

### Format File .lin
File `.lin` menggunakan format pipe-delimited dengan tag-tag berikut:
- `vg|...` — Info turnamen/event
- `pn|...` — Nama pemain (urutan: S, W, N, E)
- `rs|...` — Hasil semua board (contoh: `4HE=` artinya 4 Heart oleh East, making)
- `qx|o1|` atau `qx|c1|` — Penanda board (o=open room, c=closed room)
- `st|...` — Settings
- `md|DEALER|SOUTH_HAND|WEST_HAND|NORTH_HAND|EAST_HAND` — Distribusi kartu
  - Dealer: 1=S, 2=W, 3=N, 4=E
  - Format hand: `S<spades>H<hearts>D<diamonds>C<clubs>` (T=10, J,Q,K,A)
  - East hand kadang kosong (bisa dihitung dari sisa kartu)
- `sv|...` — Vulnerability (o=none, n=NS, e=EW, b=both)
- `mb|...` — Bidding (p=pass, d=double, r=redouble, 1C,1D,1H,1S,1N,...,7N)
- `an|...` — Annotation/alert pada bid
- `pc|...` — Card play (contoh: `cQ` = Queen of clubs, `hA` = Ace of hearts)
- `mc|N|` — Claim N tricks
- `pg||` — Separator

### Contoh Parsing Hand
`md|3SJ97HJTDAQ6543CQ6,SKQ8543HK987DTCAK,SAT62HAQD82CT8732,SH65432DKJ97CJ954`
- Dealer: 3 (North)
- South: ♠J97 ♥JT ♦AQ6543 ♣Q6
- West: ♠KQ8543 ♥K987 ♦T ♣AK
- North: ♠AT62 ♥AQ ♦82 ♣T8732
- East: ♠- ♥65432 ♦KJ97 ♣J954

### Contoh Parsing Bidding
`mb|p|mb|p|mb|1D|mb|d|mb|1H|mb|p|mb|1S|mb|p|mb|1N|mb|2H|mb|p|mb|4H|mb|p|mb|p|mb|p|`
Urutan bidding dimulai dari dealer, lalu searah jarum jam:
- North: pass → 1D → 1S → 1N → pass → pass
- East: pass → double → pass → 2H → 4H → pass
- South: (sesuai urutan) ...
- West: (sesuai urutan) ...
Kontrak final: 4H

### Contoh Parsing Hasil
`rs|4HE=,4HW=,...` → Board 1: 4 Hearts oleh East making exactly, dst.
Format: `<level><suit><declarer><result>` dimana result: `=` (making), `+N` (overtrick), `-N` (down)

---

## Referensi Paper Utama

### 1. C23 — Lin et al. (2023) — `C23Bridge_bid.pdf`
**"Two-stage Random Forest for Non-competitive Bidding"**
- **Metode**: 2-stage Random Forest (stage 1: prediksi suit, stage 2: prediksi game category)
- **Dataset**: ~107,000 instances dari BBO "Just Declare"
- **Fitur utama**:
  - HCP per suit (player): 0-10 per suit
  - Total HCP range (lower/upper): 0-40
  - Jumlah kartu per suit (lower/upper): 0-13
  - Balanced label (player & partner): 0=balanced, 1=likely balanced, 2=likely unbalanced, 3=unbalanced
  - Stopper per suit: 0=no stopper, 1=unknown, 2=partial-stopper (QXX), 3=stopper (AJX)
  - Vulnerability: 1=none, 2=NS, 3=EW, 4=both
  - Bidding history: 72-bit one-hot encoding
- **Label/Target**: Kontrak terbaik ditentukan via DDS (shuffle 100 EW hands, ambil kontrak skor tertinggi)
- **Game categories**: partial game, game, slam, grand slam
- **Hasil**: Akurasi Same Category (SC) = 0.773, IMP gain vs BBO = 0.212 IMP/board
- **Evaluasi**: 5-fold cross-validation × 10 runs
- **7 Indikator evaluasi**: MS, SCA, SCU, SSE, O, SS, SC

### 2. C25 — Chen & Yang (2025) — `C25Bridge_Bid_competitive.pdf`
**"Neural Network for Competitive Bidding"**
- **Metode**: Neural network (612-1224-612-37 architecture)
- **Dataset**: ~490,000 boards dari BBO (competitive bidding)
- **Feature encoding terbaik**: Feature extraction dengan H+TL+NL dan H+Lwh (akurasi 0.87)
- **Bidding encoding**: Custom method preserving order + context (akurasi 0.87 vs label encoding 0.66)
- **Fitur**:
  - Hand features: HCP (Goren atau H+Lwh formula), jumlah kartu per suit, honor cards
  - Bidding state: 4×141-bit vector per pemain (OWN, LHO, PD, RHO)
  - Bidding explanation: lower/upper bound per suit + HCP range
- **Hasil**: Akurasi test = 0.87, IMP: win 6.4%, tie 87.6%, loss 6%
- **Evaluasi**: Stratified 10-fold cross-validation × 10 runs

### 3. Mathematics 2022 — `mathematics1003187.pdf`
**"Bridge Bidding AI Supporting Multiple Bidding Systems"**
- Dual neural network: Bid Selection Model + Evaluation Model
- Encoding kartu: High cards placeholder, small cards (2-9) count-based
- Mendukung multiple bidding systems (precision, CCBA)

### 4. Alleviating Local Optima — `Alleviating_Local_Optima_in_Bridge_Bidding_via_Diverse_PPO_Ensembling.pdf`
- PPO-based reinforcement learning + ensemble learning
- Search pruning module untuk filter aksi
- Terlalu kompleks untuk skripsi S1, tapi bisa direferensikan

### 5. BridgeHand2Vec — `2310_06624v1.pdf`
- Representasi vektor untuk hand bridge
- Neural network untuk estimasi tricks (DDBP)
- Bisa direferensikan untuk bagian related work

---

## Arsitektur Model yang Akan Dibangun

### Pipeline Lengkap:
```
File .lin → Parser → Raw Dataset (CSV)
    → Feature Extraction → Feature Dataset
    → DDS Labeling → Labeled Dataset
    → 2-Stage Random Forest → Prediksi Kontrak
    → Evaluasi (Akurasi + IMP)
```

### Stage 1: Prediksi Suit
- **Input**: Fitur hand + bidding
- **Output**: Club, Diamond, Heart, Spade, NoTrump (5 kelas)
- **Model**: RandomForestClassifier

### Stage 2: Prediksi Game Category
- **Input**: Fitur hand + bidding + hasil prediksi suit
- **Output**: Partial Game, Game, Slam, Grand Slam (4 kelas)
- **Model**: RandomForestClassifier
- **Penyesuaian**: Jika suit = minor dan category = game, level minimum = 5 (bukan 4)

### Kontrak Final
Kombinasi suit + category → kontrak spesifik (contoh: Heart + Game = 4H)

---

## Fitur yang Akan Diekstrak

### A. Fitur Kartu Pemain (dari hand NS)
| Fitur | Deskripsi | Range |
|-------|-----------|-------|
| player_hcp_club | HCP di suit club (A=4,K=3,Q=2,J=1) | 0-10 |
| player_hcp_diamond | HCP di suit diamond | 0-10 |
| player_hcp_heart | HCP di suit heart | 0-10 |
| player_hcp_spade | HCP di suit spade | 0-10 |
| player_hcp_total | Total HCP pemain | 0-37 |
| player_num_club | Jumlah kartu club | 0-13 |
| player_num_diamond | Jumlah kartu diamond | 0-13 |
| player_num_heart | Jumlah kartu heart | 0-13 |
| player_num_spade | Jumlah kartu spade | 0-13 |

### B. Fitur Distribusi
| Fitur | Deskripsi | Range |
|-------|-----------|-------|
| player_balanced | 0=balanced(4333/4432/5332), 1=semi, 2=unbalanced, 3=very unbal | 0-3 |
| partner_hcp_total | Total HCP partner (dihitung langsung) | 0-37 |
| partner_num_* | Jumlah kartu per suit partner | 0-13 |
| combined_hcp | Total HCP NS combined | 0-40 |

### C. Fitur Stopper (per suit)
| Fitur | Deskripsi | Encoding |
|-------|-----------|----------|
| stopper_club | Kekuatan stopper di club | 0-3 |
| stopper_diamond | Kekuatan stopper di diamond | 0-3 |
| stopper_heart | Kekuatan stopper di heart | 0-3 |
| stopper_spade | Kekuatan stopper di spade | 0-3 |

Stopper encoding: 0=no stopper (xxx), 1=unknown, 2=partial (Qxx), 3=stopper (AJx, Kxx+)

### D. Fitur Situasi
| Fitur | Deskripsi | Encoding |
|-------|-----------|----------|
| vulnerability | Kondisi vulnerability | 0-3 (o/n/e/b) |
| dealer | Posisi dealer | 0-3 (S/W/N/E) |

### E. Fitur Bidding History
| Fitur | Deskripsi |
|-------|-----------|
| bidding_history | One-hot encoding urutan bidding (72-bit atau 141-bit per pemain) |

---

## Tools & Library yang Digunakan
- **Python 3.x**
- **pandas** — Manipulasi data/dataframe
- **scikit-learn** — RandomForestClassifier, cross-validation, metrics
- **numpy** — Operasi numerik
- **matplotlib / seaborn** — Visualisasi (confusion matrix, feature importance)
- **python-dds** atau **redeal** — Double Dummy Solver (untuk labeling)
- **Jupyter Notebook** — Eksperimen interaktif (opsional)

---

## Struktur Direktori yang Direncanakan
```
bridge-skripsi/
├── CLAUDE.md              # File ini
├── README.md              # Panduan proyek
├── data/
│   ├── raw/               # File .lin mentah
│   ├── parsed/            # Hasil parsing (CSV)
│   └── processed/         # Dataset final (fitur + label)
├── src/
│   ├── parser.py          # Parser file .lin
│   ├── features.py        # Feature extraction
│   ├── labeling.py        # DDS labeling
│   ├── model.py           # Training Random Forest
│   └── evaluate.py        # Evaluasi model
├── notebooks/
│   └── exploration.ipynb  # Eksplorasi data
├── results/
│   ├── figures/           # Grafik dan visualisasi
│   └── metrics/           # Hasil evaluasi
└── docs/
    └── skripsi/           # Draft bab skripsi
```

---

## Tahapan Pengerjaan (Roadmap)

### Minggu 1-2: Parsing Data .lin
- [ ] Buat parser untuk file .lin → CSV
- [ ] Ekstrak: hand 4 pemain, vulnerability, dealer, bidding sequence, kontrak final, hasil
- [ ] Validasi hasil parsing (bandingkan manual dengan output)
- [ ] Kumpulkan lebih banyak file .lin dari BBO

### Minggu 3-4: Feature Extraction
- [ ] Hitung HCP per suit untuk setiap pemain
- [ ] Hitung distribusi (jumlah kartu per suit)
- [ ] Tentukan balanced/unbalanced label
- [ ] Hitung stopper per suit
- [ ] Encode bidding history (one-hot 72-bit)
- [ ] Gabungkan semua fitur ke satu dataframe

### Minggu 5-6: Labeling dengan DDS
- [ ] Install/setup Double Dummy Solver
- [ ] Untuk setiap NS hand, shuffle 100 EW hands
- [ ] Jalankan DDS untuk setiap shuffle → catat kontrak + tricks
- [ ] Tentukan kontrak terbaik (Most Suitable) berdasarkan frekuensi skor tertinggi
- [ ] Buat label: suit target + game category target

### Minggu 7-8: Training Model
- [ ] Split dataset (atau gunakan k-fold cross-validation)
- [ ] Train Random Forest Stage 1 (prediksi suit): Club/Diamond/Heart/Spade/NT
- [ ] Train Random Forest Stage 2 (prediksi category): Partial/Game/Slam/GrandSlam
- [ ] Tune hyperparameter (n_estimators, max_depth, dll)
- [ ] Gabungkan prediksi → kontrak final

### Minggu 9-10: Evaluasi
- [ ] Hitung akurasi per stage dan overall
- [ ] Buat confusion matrix
- [ ] Hitung 7 indikator: MS, SCA, SCU, SSE, O, SS, SC
- [ ] Hitung IMP score (bandingkan dengan kontrak BBO)
- [ ] Analisis feature importance
- [ ] Visualisasi hasil

### Minggu 11-14: Penulisan Skripsi
- [ ] Bab 1: Pendahuluan (latar belakang, rumusan masalah, tujuan)
- [ ] Bab 2: Tinjauan Pustaka (bridge, ML, Random Forest, related work)
- [ ] Bab 3: Metodologi (pipeline, fitur, model, evaluasi)
- [ ] Bab 4: Hasil dan Pembahasan (akurasi, confusion matrix, IMP, feature importance)
- [ ] Bab 5: Kesimpulan dan Saran
- [ ] Revisi dan persiapan sidang

---

## Catatan Penting untuk Claude Code

### Saat Parsing:
- East hand di `.lin` kadang kosong → hitung dari sisa 52 kartu dikurangi S+W+N
- Huruf kecil di `pc|` tag: s=spade, h=heart, d=diamond, c=club
- Huruf besar di `md|` tag: S=spade, H=heart, D=diamond, C=club
- T = 10 (Ten), bukan Trump
- `mb|p|` = pass, `mb|d|` = double, `mb|r|` = redouble
- `an|...|` setelah `mb|` adalah annotation/alert, bukan bid

### Saat Feature Extraction:
- HCP Goren system: A=4, K=3, Q=2, J=1
- Balanced: 4333, 4432, 5332 (dan 5422 menurut beberapa referensi)
- Pastikan fitur dihitung dari perspektif NS (North-South partnership)

### Saat Training:
- Gunakan stratified k-fold cross-validation (5 atau 10 fold)
- Ulangi cross-validation 10 kali untuk stabilitas
- Random Forest default scikit-learn sudah cukup baik sebagai baseline
- Feature importance dari Random Forest berguna untuk analisis di Bab 4

### Saat Evaluasi:
- Same Category (SC) adalah metrik utama (lebih penting dari Most Suitable)
- IMP scoring: hitung selisih skor kontrak prediksi vs kontrak BBO, konversi ke IMP table
- Game categories dan level mapping:
  - Partial game: 1-level sampai 3-level (kecuali 3NT)
  - Game: 3NT, 4H, 4S, 5C, 5D
  - Slam: 6-level
  - Grand Slam: 7-level
