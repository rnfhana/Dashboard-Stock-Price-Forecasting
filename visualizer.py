import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st

# Palet Warna ESG
COLORS = {'forest': '#1B4332', 'emerald': '#2D6A4F', 'mint': '#D8F3DC', 'actual': '#1B4332', 'pred': '#E76F51', 'warning': '#E76F51'}

def show_variable_reference():
    st.markdown("### 📋 Referensi Dataframe (Yt+3 Target)")
    data = {
        "Kode": ["Yt+3", "Yt", "X1-X3", "X4-X7", "X8-X11"],
        "Nama": ["Target Forecast", "Close Price", "Price Data", "Technical", "Sentiment"],
        "Deskripsi": ["Harga 3 Hari ke Depan", "Harga Penutupan Saat Ini", "Open, High, Low", "Market Activity", "ESG Data"]
    }
    st.table(pd.DataFrame(data))

def plot_line_history(df):
    if isinstance(df, dict): df = pd.DataFrame(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'] if 'date' in df.columns else df.index, 
                             y=df['Yt'], name='Close Price', line=dict(color=COLORS['forest'])))
    fig.update_layout(title="Historical Price Trend", template="plotly_white", height=350)
    return fig

def plot_sentiment_metrics(df):
    if isinstance(df, dict): df = pd.DataFrame(df)
    pos_sum = df['X10'].sum()
    neg_sum = df['X11'].sum()
    fig_pie = px.pie(values=[pos_sum, neg_sum], names=['Positive Events', 'Negative Events'], 
                     color_discrete_sequence=[COLORS['emerald'], COLORS['warning']], title="Total Sentiment Distribution")
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=df['X8'], name='AVG Pos', marker_color=COLORS['emerald'], opacity=0.6))
    fig_hist.add_trace(go.Histogram(x=df['X9'], name='AVG Neg', marker_color=COLORS['warning'], opacity=0.6))
    fig_hist.update_layout(barmode='overlay', title="Sentiment Score Distribution (X8 & X9)", template="plotly_white")
    return fig_pie, fig_hist

def plot_loss_curve(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history['loss'], name='Train Loss', line=dict(color=COLORS['emerald'])))
    fig.add_trace(go.Scatter(y=history['val_loss'], name='Val Loss', line=dict(color=COLORS['warning'])))
    fig.update_layout(title="Model Learning Curve (Loss)", template="plotly_white")
    return fig

def plot_actual_vs_pred(actual, pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual, name='Actual Yt+3', line=dict(color=COLORS['actual'])))
    fig.add_trace(go.Scatter(y=pred, name='Predicted Yt+3', line=dict(color=COLORS['pred'], dash='dash')))
    fig.update_layout(title="Performance Evaluation (Actual vs Predicted Yt+3)", template="plotly_white")
    return fig

def plot_forecast_only(df_fut):
    if isinstance(df_fut, dict): df_fut = pd.DataFrame(df_fut)
    fig = go.Figure()
    y_val = df_fut['prediction'] if 'prediction' in df_fut.columns else df_fut.iloc[:,0]
    fig.add_trace(go.Scatter(y=y_val, mode='lines+markers', name='3-Day Forecast', line=dict(color=COLORS['emerald'], dash='dot')))
    fig.update_layout(title="3-Day Future Price Forecast", template="plotly_white")
    return fig