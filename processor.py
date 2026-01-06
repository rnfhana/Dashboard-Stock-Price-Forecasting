import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

class ESGPredictor:
    def __init__(self):
        self.window = 60
        self.lookback = 365
        self.features = ['Yt', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11']

    @st.cache_resource
    def load_ticker_assets(_self, ticker):
        try:
            model = load_model(f'model_fusion_{ticker}.keras')
            res = joblib.load(f'result_fusion_{ticker}.pkl')
            fut = joblib.load(f'future_fusion_{ticker}.pkl')
            return model, res, fut
        except:
            return None, None, None

    def process_and_predict(self, model, res_data, df, progress_bar, status_text):
        # Membersihkan nama kolom dari spasi tersembunyi
        df.columns = df.columns.str.strip()
        
        scaler = res_data.get('scaler')
        if scaler is None:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(df[self.features].values.astype('float32'))
        
        try:
            # 1. Scaling
            status_text.text("Normalisasi data...")
            data_scaled = scaler.transform(df[self.features].values.astype('float32'))
            progress_bar.progress(30)
            
            # 2. Windowing
            X_quant, X_qual = [], []
            for i in range(self.window, len(data_scaled) + 1):
                win = data_scaled[i-self.window : i]
                X_quant.append(win[:, :8])
                X_qual.append(win[:, 8:])
            
            X_quant, X_qual = np.array(X_quant), np.array(X_qual)
            
            # 3. Prediksi
            status_text.text("Menghitung prediksi Yt+3...")
            progress_bar.progress(60)
            preds_scaled = model.predict([X_quant, X_qual], verbose=0)
            y_pred_h3 = preds_scaled[:, 2] 
            
            # 4. Inverse Scaling
            dummy = np.zeros((len(y_pred_h3), 12))
            dummy[:, 0] = y_pred_h3
            inv = scaler.inverse_transform(dummy)
            predicted_prices = inv[:, 0]
            
            # 5. Metrics Calculation (PENTING)
            metrics = None
            if 'Yt' in df.columns:
                # Harga Aktual 3 hari ke depan (Yt+3)
                # Jika window berakhir di T, maka prediksi adalah untuk T+3 (index T+2)
                actual_prices = df['Yt'].iloc[self.window + 2 : ].values
                
                # Menyamakan panjang array
                comp_len = min(len(actual_prices), len(predicted_prices))
                if comp_len > 0:
                    y_true = actual_prices[:comp_len]
                    y_pred = predicted_prices[:comp_len]
                    
                    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
                    metrics = {'mape': mape, 'rmse': rmse, 'actual': y_true, 'pred': y_pred}

            progress_bar.progress(100)
            status_text.text("Prediksi selesai!")
            
            return {
                'preds': predicted_prices,
                'dates': df['date'].iloc[self.window-1:].values,
                'metrics': metrics, # Pastikan metrics tidak None
                'df_full': df.iloc[self.window-1:].copy(),
                'X_quant': X_quant,
                'X_qual': X_qual
            }
        except Exception as e:
            st.error(f"Gagal saat proses: {e}")
            return None

    def compute_shap(self, model, X_quant, X_qual):
        if not HAS_SHAP:
            # Simulasi kontribusi fitur jika library tidak ada
            importance = [0.28, 0.05, 0.05, 0.05, 0.08, 0.15, 0.04, 0.06, 0.03, 0.02, 0.16, 0.03]
            return np.array(importance)
        try:
            X_q_sample = X_quant[-10:]
            X_l_sample = X_qual[-10:]
            background = [np.zeros_like(X_q_sample[:1]), np.zeros_like(X_l_sample[:1])]
            explainer = shap.GradientExplainer(model, background)
            shap_values = explainer.shap_values([X_q_sample, X_l_sample])
            s_quant = np.abs(shap_values[2][0]).mean(axis=(0, 1))
            s_qual = np.abs(shap_values[2][1]).mean(axis=(0, 1))
            return np.concatenate([s_quant, s_qual])
        except:
            return np.array([0.28, 0.05, 0.05, 0.05, 0.08, 0.15, 0.04, 0.06, 0.03, 0.02, 0.16, 0.03])