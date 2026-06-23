# Perubahan Algoritma: Random Forest → MLP & LSTM

## 📋 Ringkasan Perubahan

Proyek Bridge bidding prediction telah diubah dari menggunakan **Random Forest** menjadi **Neural Network** dengan dua pilihan arsitektur:

### 1. **MLP (Multi-Layer Perceptron)**

- **Cocok untuk:** Tabular features (data tabell dengan fitur independen)
- **Arsitektur:** Input (98) → Dense(256) + Dropout(0.3) → Dense(128) + Dropout(0.3) → Dense(64) + Dropout(0.3) → Output
- **Kelebihan:**
  - Training lebih cepat
  - Parameter lebih sedikit
  - Lebih interpretable
  - Cocok untuk data tabell seperti bridge features

### 2. **LSTM (Long Short-Term Memory)**

- **Cocok untuk:** Sequence/temporal data
- **Arsitektur:** Input (reshaped ke sequence) → LSTM(128) + Dropout(0.3) → LSTM(64) + Dropout(0.3) → Dense(64) + Dropout(0.2) → Output
- **Kelebihan:**
  - Capture temporal dependencies
  - Baik untuk sequential patterns
  - Features direshape menjadi 10 timesteps × 9 features per timestep
- **Catatan:** Features tabular direshape secara artificial untuk kompatibilitas LSTM

---

## 🔧 File yang Diubah

### 1. `requirements.txt`

Ditambahkan dependencies untuk deep learning:

```
tensorflow>=2.13.0
keras>=2.13.0
```

### 2. `src/model.py`

- ❌ Dihapus: Class `TwoStageRF`, `RF_PARAMS`
- ✅ Ditambahkan: Class `TwoStageMLP`, Class `TwoStageLSTM`
- ✅ Diubah: Function `train()` sekarang support parameter `model_type`
- ✅ Diubah: Function `save_model()` dan `load_model()` untuk Keras models
- ✅ Ditambahkan: `StandardScaler` untuk normalisasi features
- ✅ Ditambahkan: `LabelEncoder` untuk encoding labels

### 3. `notebooks/06_training.ipynb`

- ✅ Updated markdown untuk menjelaskan MLP & LSTM
- ✅ Updated cell training untuk menggunakan `model_type` parameter
- ✅ Ditambahkan `MODEL_TYPE` variable untuk switch antara MLP dan LSTM

---

## 🚀 Cara Menggunakan

### Setup Dependencies

```bash
pip install -r requirements.txt
```

### Training Model

Di notebook `06_training.ipynb`, ubah `MODEL_TYPE` sesuai kebutuhan:

#### Menggunakan MLP (Rekomendasi untuk dataset tabular)

```python
MODEL_TYPE = "mlp"
model = train(X_train, y_suit_train, y_cat_train, model_type=MODEL_TYPE)
```

#### Menggunakan LSTM

```python
MODEL_TYPE = "lstm"
model = train(X_train, y_suit_train, y_cat_train, model_type=MODEL_TYPE)
```

---

## 📊 Model Architecture Comparison

| Aspek            | Random Forest               | MLP                   | LSTM                           |
| ---------------- | --------------------------- | --------------------- | ------------------------------ |
| Input            | 98 features                 | 98 features           | 98 features (reshaped to 10×9) |
| Training Speed   | Cepat                       | Sedang                | Lambat                         |
| Memory           | Rendah-Sedang               | Sedang                | Tinggi                         |
| Interpretability | Tinggi (feature importance) | Rendah                | Rendah                         |
| Regularization   | Tree depth                  | Dropout               | Dropout + LSTM gates           |
| Cross-validation | Langsung supported          | Manual atau callbacks | Manual atau callbacks          |
| Hyperparameter   | n_estimators, depth         | hidden_units, epochs  | lstm_units, seq_length         |

---

## 🔑 Default Hyperparameters

### MLP/LSTM (NN_PARAMS)

```python
NN_PARAMS = {
    "epochs": 100,
    "batch_size": 32,
    "validation_split": 0.2,
    "verbose": 0,
}
```

### MLP Architecture

```python
hidden_units = [256, 128, 64]
```

### LSTM Architecture

```python
lstm_units = [128, 64]
seq_length = 10
```

Untuk override, pass `params` dictionary ke function `train()`:

```python
custom_params = {
    "epochs": 200,
    "batch_size": 16,
    "validation_split": 0.2,
    "verbose": 1
}
model = train(X_train, y_suit_train, y_cat_train,
              model_type="mlp", params=custom_params)
```

---

## 📂 File Output

### Model Files

- `results/metrics/model_suit.h5` — MLP/LSTM untuk prediksi suit
- `results/metrics/model_category.h5` — MLP/LSTM untuk prediksi category
- `results/metrics/model_suit_lstm.h5` — LSTM suit model (jika menggunakan LSTM)
- `results/metrics/model_category_lstm.h5` — LSTM category model (jika menggunakan LSTM)

### Supporting Files

- `results/metrics/scaler.pkl` — StandardScaler untuk normalisasi
- `results/metrics/encoder_suit.pkl` — LabelEncoder untuk suit labels
- `results/metrics/encoder_category.pkl` — LabelEncoder untuk category labels

---

## 💡 Rekomendasi

1. **Mulai dengan MLP** - Lebih cepat, lebih interpretable, cocok untuk tabular bridge features
2. **Gunakan LSTM** - Jika ada temporal pattern dalam bidding sequence yang ingin dikapture
3. **Validation Split** - Default 20% untuk validation, dapat diubah di `NN_PARAMS`
4. **Epochs** - Default 100, increase jika model belum converge
5. **Batch Size** - Default 32, reduce jika GPU memory terbatas

---

## 🔄 Migrasi dari Random Forest

Jika ada code yang menggunakan `TwoStageRF`, lakukan update:

```python
# Lama (Random Forest)
from model import TwoStageRF, train, save_model
model = train(X_train, y_suit_train, y_cat_train)

# Baru (MLP - Rekomendasi)
from model import TwoStageMLP, train, save_model
model = train(X_train, y_suit_train, y_cat_train, model_type="mlp")

# Baru (LSTM)
from model import TwoStageLSTM, train, save_model
model = train(X_train, y_suit_train, y_cat_train, model_type="lstm")
```

---

## ⚠️ Catatan Penting

1. **Cross-Validation:** Tidak ada built-in cross-validation untuk neural network. Untuk CV, perlu manual loop atau use Keras `KFold`.
2. **Feature Importance:** Neural network tidak memiliki feature importance seperti Random Forest. Gunakan SHAP atau other explainability tools.
3. **Reproducibility:** Set TensorFlow random seed di notebook jika diperlukan:
   ```python
   import tensorflow as tf
   tf.random.set_seed(42)
   ```
4. **GPU:** Jika GPU tersedia, TensorFlow akan otomatis menggunakannya. Check dengan:
   ```python
   print(tf.config.list_physical_devices('GPU'))
   ```

---

## 📝 Next Steps

- [ ] Run `06_training.ipynb` dengan MLP
- [ ] Evaluate model di `07_evaluasi.ipynb`
- [ ] Try LSTM jika diperlukan
- [ ] Hyperparameter tuning untuk improve performance
- [ ] Compare MLP vs LSTM results
