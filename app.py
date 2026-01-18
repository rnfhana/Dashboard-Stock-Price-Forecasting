import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from style_utils import apply_custom_style
from visualizer import *
from processor import ESGPredictor

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Stock Price Forecasting", layout="wide")
apply_custom_style()
engine = ESGPredictor()

# 2. Sidebar
st.sidebar.title("🍃Stock Price Forecasting")
st.sidebar.caption("IDX ESG Leaders")
ticker = st.sidebar.radio("Pilih Emiten", ["AKRA", "BBRI", "BMRI", "PGAS", "UNVR"])

# 3. Load Model & Data Awal
model, res_data, fut_data = engine.load_ticker_assets(ticker)

# 4. LOGIKA GLOBAL
use_csv = 'bulk_results' in st.session_state and st.session_state.get('current_ticker') == ticker
active_res = st.session_state.get('bulk_results') if use_csv else None

# 5. Struktur Tab
tab1, tab2, tab3, tab4 = st.tabs(["📂 Upload CSV", "📊 Characteristics", "🧬 Model Performance", "📈 Result"])

with tab1:
    st.info(f"💡 **Info Model:** Membutuhkan minimal {engine.lookback} hari data untuk window {engine.window} hari.")
    show_variable_reference()
    
    up_file = st.file_uploader("Upload CSV Data", type="csv")
    if up_file:
        df_uploaded = pd.read_csv(up_file)
        df_filtered = df_uploaded[df_uploaded['relevant_issuer'] == ticker].copy()
        
        if len(df_filtered) < engine.window:
            st.warning(f"Data tidak cukup. Butuh {engine.window} baris, hanya ada {len(df_filtered)}.")
        else:
            if st.button("🚀 Jalankan Prediksi Massal"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = engine.process_and_predict(model, res_data, df_filtered, progress_bar, status_text)
                
                if results:
                    st.session_state['bulk_results'] = results
                    st.session_state['current_ticker'] = ticker
                    st.success("✅ Analisis Berhasil! Silakan cek tab lainnya.")
                    st.rerun()

with tab2:
    st.subheader(f"Data Characteristics: {ticker}")
    df_char = active_res['df_full'] if use_csv else res_data.get('raw_data')
    
    if df_char is not None:
        st.plotly_chart(plot_line_history(df_char), width='stretch')
        st.write("**Statistika Deskriptif**")
        st.dataframe(df_char.describe().T)
        
        f_pie, f_hist = plot_sentiment_metrics(df_char)
        c1, c2 = st.columns(2)
        c1.plotly_chart(f_pie, width='stretch')
        c2.plotly_chart(f_hist, width='stretch')

with tab3:
    st.subheader("🧬 Model Performance Evaluation")
    
    # CEK APAKAH ADA HASIL DARI CSV
    if use_csv and active_res is not None:
        m = active_res.get('metrics')
        
        if m is not None:
            st.success("📊 Performa Model pada Data CSV yang Diunggah")
            c1, c2 = st.columns(2)
            c1.metric("MAPE (Yt+3)", f"{m['mape']:.2f}%")
            c2.metric("RMSE (Yt+3)", f"{m['rmse']:.4f}")
            
            # Plot Actual vs Prediksi dari CSV
            st.plotly_chart(plot_actual_vs_pred(m['actual'], m['pred']), width='stretch')
        else:
            st.warning("⚠️ Kolom 'Yt' ditemukan, tapi data tidak cukup untuk menghitung selisih 3 hari (Yt+3).")
    
    # TAMPILKAN PERFORMANCE DEFAULT DARI FILE .PKL
    st.markdown("---")
    st.write("📂 **Model Training Performance (Historical)**")
    if res_data:
        c1, c2 = st.columns(2)
        # Ambil nilai mape dari pkl, jika tidak ada tampilkan 0.00
        pkl_mape = res_data.get('metrics', {}).get('Test_MAPE_H3', 0)
        pkl_rmse = res_data.get('metrics', {}).get('Test_RMSE_H3', 0)
        
        c1.metric("Training MAPE", f"{pkl_mape:.2f}%")
        c2.metric("Training RMSE", f"{pkl_rmse:.4f}")
        
        if 'history' in res_data:
            st.plotly_chart(plot_loss_curve(res_data['history']), width='stretch')

# --- TAB 4: RESULT & REKOMENDASI (Yt+3) ---
with tab4:
    st.subheader("📈 Prediction Result & Explainable AI (SHAP)")
    
    if use_csv and active_res is not None:
        # 1. Grafik Prediksi Harga dari CSV
        target_dates = pd.to_datetime(active_res['dates']) + pd.Timedelta(days=3)
        fig_csv = go.Figure()
        fig_csv.add_trace(go.Scatter(x=target_dates, y=active_res['preds'], 
                                     name="Prediction Yt+3", line=dict(color='#2D6A4F', width=2)))
        fig_csv.update_layout(title="Hasil Prediksi Harga Target (Yt+3)", xaxis_title="Tanggal Target")
        st.plotly_chart(fig_csv, width='stretch')
        
        # 2. Fitur SHAP (Explainable AI)
        st.markdown("---")
        st.write("### 🧠 Global Feature Importance (SHAP)")
        
        with st.spinner("Menganalisis kontribusi fitur..."):
            importance = engine.compute_shap(model, active_res['X_quant'], active_res['X_qual'])
            
            if importance is not None:
                feat_names = ['Yt','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']
                shap_df = pd.DataFrame({'Feature': feat_names, 'Importance': importance}).sort_values('Importance')
                
                fig_s = px.bar(shap_df, x='Importance', y='Feature', orientation='h', 
                               color='Importance', color_continuous_scale='Greens')
                st.plotly_chart(fig_s, width='stretch')
                
                # Cek apakah pakai library asli atau simulasi
                try:
                    import shap
                except ImportError:
                    st.caption("ℹ️ Menampilkan estimasi kontribusi fitur (Mode Penyelamatan).")
            else:
                st.error("Analisis XAI tidak tersedia.")

        # 3. Strategi & Rekomendasi Investor (WAJIB ADA DI DALAM BLOK IF)
        st.markdown("---")
        st.subheader("💡 Strategi & Rekomendasi Investor")
        
        # Pastikan ada data prediksi sebelum menghitung
        if len(active_res['preds']) >= 2:
            last_pred = active_res['preds'][-1]
            prev_pred = active_res['preds'][-2]
            
            # Hindari pembagian dengan nol
            if prev_pred != 0:
                change_pct = ((last_pred - prev_pred) / prev_pred) * 100
            else:
                change_pct = 0
            
            col_rec1, col_rec2 = st.columns([1, 2])
            
            with col_rec1:
                if change_pct > 0.5:
                    st.success(f"### 🟢 Rekomendasi: **BUY**")
                    st.metric("Potensi Kenaikan", f"+{change_pct:.2f}%")
                elif change_pct < -0.5:
                    st.error(f"### 🔴 Rekomendasi: **SELL/WAIT**")
                    st.metric("Potensi Penurunan", f"{change_pct:.2f}%")
                else:
                    st.warning(f"### 🟡 Rekomendasi: **HOLD**")
                    st.metric("Perubahan Harga", f"{change_pct:.2f}%")

            with col_rec2:
                trend_text = 'Bullish (Naik)' if change_pct > 0 else 'Bearish (Turun)'
                st.info(f"""
                **Analisis Strategis untuk {ticker}:**
                1. **Tren:** Model memproyeksikan harga {ticker} bergerak **{trend_text}** dalam 3 hari ke depan.
                2. **Dasar Keputusan:** Rekomendasi ini didasarkan pada momentum perubahan harga prediksi $Y_{{t+3}}$ dibandingkan hari sebelumnya.
                3. **Mitigasi Risiko:** Perhatikan variabel dengan nilai SHAP tertinggi di atas sebelum mengeksekusi transaksi.
                """)
        else:
            st.warning("Data prediksi kurang dari 2 hari, tren tidak dapat dihitung.")

    else:
        # Tampilan Default (Jika belum upload CSV)
        st.plotly_chart(plot_forecast_only(fut_data), width='stretch')
        st.info("ℹ️ Silakan lakukan **Prediksi Massal** di Tab 1 untuk melihat analisis rekomendasi investor.")