import joblib
# Ganti dengan salah satu file future_fusion Anda
data = joblib.load('future_fusion_BBRI.pkl')
print("Tipe data:", type(data))
if hasattr(data, 'columns'):
    print("Daftar Kolom:", data.columns.tolist())
else:
    print("Data tidak memiliki atribut columns (mungkin isinya list atau array)")
    print(data)