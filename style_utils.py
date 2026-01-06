import streamlit as st

def apply_custom_style():
    st.markdown("""
        <style>
        /* Sidebar Background Gradient */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1B4332 0%, #2D6A4F 100%);
        }
        
        /* Sidebar Judul Utama */
        [data-testid="stSidebar"] h1 { color: white !important; margin-bottom: 0px; }

        /* Teks "IDX ESG Leaders" - DIUBAH JADI MINT */
        [data-testid="stSidebar"] .stCaption { 
            color: #D8F3DC !important; 
            font-size: 1rem; 
            font-weight: 500;
        }

        /* Label "Pilih Emiten" - Hitam Tebal, Blok Putih */
        [data-testid="stSidebar"] label[data-testid="stWidgetLabel"] {
            color: #000000 !important;
            font-weight: bold !important;
            background-color: #FFFFFF !important;
            padding: 10px 15px !important;
            border-radius: 5px !important;
            display: block !important;
            width: 100% !important;
        }

        /* Radio Button Emiten (AKRA, BBRI, dkk) - TULISAN PUTIH */
        div[data-testid="stRadio"] div[role="radiogroup"] label [data-testid="stWidgetLabel"] p,
        div[data-testid="stRadio"] div[role="radiogroup"] label p {
            color: white !important;
        }
        
        /* Menghapus background blok pada pilihan radio */
        div[data-testid="stRadio"] div[role="radiogroup"] label {
            background-color: transparent !important;
            border: none !important;
            padding: 5px 0px !important;
        }
        
        div[data-testid="stRadio"] div[role="radiogroup"] { gap: 5px; }

        /* Metric Cards */
        div.stMetric {
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 5px solid #2D6A4F;
        }
        
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #f8f9fa;
            border-radius: 5px 5px 0 0;
            padding: 10px 20px;
        }
        </style>
    """, unsafe_allow_html=True)