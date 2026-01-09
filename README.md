# Laptop Price Predictor

Panduan cepat untuk menjalankan dan mendemokan aplikasi prediksi harga laptop (Django + scikit-learn).

## 1) Persiapan
- Python 3.11+ terpasang.
- Masuk folder proyek: `cd laptop_price_project`
- Aktifkan virtual env (Windows PowerShell):
  ```ps1
  ..\.venv\Scripts\Activate.ps1
  ```
  Jika belum ada venv: `python -m venv .venv` lalu aktifkan.

## 2) Instal dependensi
```
pip install django pandas scikit-learn joblib numpy
```
(Atau `pip install -r requirements.txt` bila tersedia.)

## 3) Training model (opsional jika sudah ada model.joblib)
```
python ml/train.py
```
Hasil:
- Model: `ml/model.joblib`
- Opsi dropdown: `ml/choices.json`

## 4) Jalankan server lokal
```
python manage.py runserver
```
Akses di browser: http://127.0.0.1:8000/predict

## 5) Alur demo singkat
1. Buka halaman Predict.
2. Isi form: Brand, Type, Inches, RAM, CPU company, CPU freq, Memory, Weight, OS, (Product opsional).
3. Klik Predict → lihat hasil Euro & IDR di kartu hasil.
4. Bukti model ter-load: di terminal muncul log seperti `[predictor] Loaded model from: .../ml/model.joblib` saat server start.

## 6) Struktur utama
- `ml/train.py` : training & simpan `model.joblib`, `choices.json`.
- `predictor/runtime.py` : load artefak (model & choices).
- `predictor/views.py` : terima form, panggil model, konversi IDR, render template.
- `predictor/templates/predictor/predict.html` : form & tampilan hasil.
- `predictor/static/predictor/products.json` : daftar produk per brand/type untuk dropdown (tidak dipakai ke model).

## 7) Catatan
- Jika port 8000 sudah dipakai, hentikan proses lain atau jalankan `python manage.py runserver 127.0.0.1:8001`.
- Jika Django tidak ditemukan, pastikan venv aktif (`where python` harus menunjuk `.venv`).
- Model tidak retrain saat server jalan; cukup pastikan `ml/model.joblib` ada.
