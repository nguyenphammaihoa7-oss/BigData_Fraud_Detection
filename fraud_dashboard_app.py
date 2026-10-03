"""
Dashboard demo giam sat gian lan the tin dung theo thoi gian (mo phong), xay dung
bang Streamlit, tai lai cac model da duoc huan luyen va luu boi
train_and_compare_models.py (thu muc models/ voi 5 model .joblib, scaler.joblib,
manifest.json).

Day la mo phong (simulation), khong phai luong du lieu Kafka/MongoDB that: script
phat lai, tung giao dich mot theo thoi gian, cac giao dich lay tu TAP TEST da giu
rieng luc huan luyen (56,746 giao dich model chua tung thay), nen nhan du doan
hien thi la du doan that cua model, khong phai dan dung. Ty le gian lan xuat hien
trong luong demo co the chinh bang thanh truot de de quan sat (ty le that trong
du lieu chi 0.17%).

Chay (terminal, dung trong thu muc chua file nay):
    streamlit run fraud_dashboard_app.py

Thu vien can cai (ngoai cac thu vien da cai o buoc huan luyen model):
    pip install streamlit plotly
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier  # noqa: F401 (can thiet de unpickle joblib)
from sklearn.linear_model import LogisticRegression  # noqa: F401
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier  # noqa: F401
from sklearn.preprocessing import StandardScaler  # noqa: F401

# ============================== CONFIG ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RANDOM_STATE = 42
TEST_SIZE = 0.2
HISTORY_DISPLAY_ROWS = 12          # so dong hien thi trong bang luong giao dich
ROLLING_WINDOW = 60                 # so tick gan nhat dung de ve bieu do ty le gian lan truot
PROJECT_NAME = "Xây dựng hệ thống Big Data phát hiện gian lận thẻ tín dụng theo thời gian thực dựa trên mô hình AI"
# ======================================================================

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
    "LightGBM": "lightgbm.joblib",
    "MLP + SMOTE": "mlp_smote.joblib",
}
BASE_MODEL_ORDER = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "MLP + SMOTE"]
ALL_MODEL_ORDER = BASE_MODEL_ORDER + ["Voting Ensemble"]

COLOR_LEGIT = "#2E75B6"
COLOR_FRAUD = "#C00000"
MODEL_COLORS = {
    "Logistic Regression": "#4C72B0", "Random Forest": "#55A868", "XGBoost": "#C44E52",
    "LightGBM": "#8172B2", "MLP + SMOTE": "#CCB974", "Voting Ensemble": "#111111",
}

st.set_page_config(page_title=PROJECT_NAME, layout="wide", page_icon="🛡️")


# ----------------------------------------------------------------------
# CSS tuy chinh giao dien (co ban sang / toi)
# ----------------------------------------------------------------------
def render_css(dark: bool):
    # Chu dao: trong che do sang TAT CA chu la mau den thuan, trong che do toi TAT CA chu la
    # mau trang thuan; nhan/phu de van dung den/trang nhung giam do dam (opacity) de tao phan cap
    # thay vi doi sang mot mau khac (xanh xam nhu truoc).
    if dark:
        app_bg, app_text = "#0b1220", "#ffffff"
        card_bg, card_border = "#141c2e", "#2c3b57"
        label_color, sub_color = "rgba(255,255,255,0.62)", "rgba(255,255,255,0.48)"
        value_color = "#ffffff"
        table_border, table_head = "#2c3b57", "rgba(255,255,255,0.62)"
        row_hover = "#1b2740"
        glow_from, glow_to = "#2a3a1f", "transparent"
        badge_safe_bg, badge_safe_text = "rgba(39,174,96,0.18)", "#6fe3a0"
        badge_warn_bg, badge_warn_text = "rgba(255,193,7,0.18)", "#ffd166"
        badge_danger_bg, badge_danger_text = "rgba(192,0,0,0.22)", "#ff8a8a"
        sidebar_bg = "#0f1726"
    else:
        app_bg, app_text = "#f5f7fa", "#000000"
        card_bg, card_border = "#ffffff", "#dde2e8"
        label_color, sub_color = "rgba(0,0,0,0.6)", "rgba(0,0,0,0.45)"
        value_color = "#000000"
        table_border, table_head = "#e2e6eb", "rgba(0,0,0,0.6)"
        row_hover = "#f8fafc"
        glow_from, glow_to = "#fff6cc", "transparent"
        badge_safe_bg, badge_safe_text = "#e4f3e9", "#1b7a43"
        badge_warn_bg, badge_warn_text = "#fff3cd", "#92650a"
        badge_danger_bg, badge_danger_text = "#fde2e2", "#c00000"
        sidebar_bg = "#ffffff"
    btn_bg, btn_text, btn_border = ("#1b2740", "#ffffff", "#2e3f5c") if dark else ("#ffffff", "#000000", "#d6dde6")
    btn_hover_border = "#2e75b6"
    FONT_STACK = "'Times New Roman', Times, 'Liberation Serif', serif"

    st.markdown(f"""
    <style>
    html, body, [class*="css"] {{ font-family: {FONT_STACK}; }}
    .stApp {{ background-color: {app_bg}; color: {app_text}; font-family: {FONT_STACK}; }}
    section[data-testid="stSidebar"] {{ background-color: {sidebar_bg}; }}
    .stApp, section[data-testid="stSidebar"] * {{ color: {app_text}; font-family: {FONT_STACK}; }}
    /* Ap dung Times New Roman cho chu binh thuong nhung KHONG dong vao cac phan tu icon
       (mui ten thu gon sidebar, icon trong o chon, ...) vi chung dung font icon-ligature rieng,
       ep font chu se lam hien chu thuong thay vi hien icon */
    .stApp p, .stApp span, .stApp div, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp h5, .stApp h6, .stApp li, .stApp td, .stApp th, .stApp button, .stApp input,
    .stApp textarea, .stApp table,
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div, section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] button, section[data-testid="stSidebar"] input {{
        font-family: {FONT_STACK} !important;
    }}
    .stApp span[data-testid="stIconMaterial"], section[data-testid="stSidebar"] span[data-testid="stIconMaterial"],
    .stApp span[class*="material-symbols"], .stApp span[class*="material-icons"] {{
        font-family: 'Material Symbols Rounded' !important;
    }}

    /* Nen bo khoang trang mac dinh cua Streamlit giua cac khoi, nhung van chua du cao
       de khong bi thanh toolbar/status (Running.../Stop) cua Streamlit de len tren */
    div.block-container {{ padding-top: 3.6rem; padding-bottom: 1.2rem; max-width: 1500px; }}
    [data-testid="stVerticalBlock"] {{ gap: 0.8rem; }}
    [data-testid="stHorizontalBlock"] {{ gap: 0.8rem; }}
    h5, .stMarkdown h5 {{ margin-top: 0.1rem; margin-bottom: 0.35rem; }}

    /* Thanh header/toolbar tren cung (noi hien chu "Running..."/nut Stop) dong bo theme,
       khong de trang/mo nhat nua */
    header[data-testid="stHeader"] {{ background-color: {app_bg} !important; }}
    [data-testid="stToolbar"] {{ color: {app_text} !important; }}
    [data-testid="stStatusWidget"] {{
        background-color: {card_bg} !important; color: {app_text} !important;
        border: 1px solid {card_border} !important; border-radius: 8px !important;
    }}
    [data-testid="stStatusWidget"] * {{ color: {app_text} !important; }}
    [data-testid="stStatusWidget"] svg {{ fill: {app_text} !important; }}

    .section-label {{
        font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
        color: {label_color}; margin: 2px 0 6px 0;
    }}

    .project-title {{
        display: flex; align-items: center; gap: 16px; font-size: 30px; font-weight: 700;
        color: {value_color}; margin: 0 0 12px 0; padding-bottom: 12px;
        border-bottom: 1px solid {card_border}; line-height: 1.25;
    }}
    .project-title .proj-badge {{
        font-size: 21px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em;
        background: linear-gradient(135deg, #0f2d52, #2e75b6); color: #ffffff;
        padding: 6px 18px; border-radius: 999px; flex-shrink: 0;
    }}

    .hero-banner {{
        background: linear-gradient(135deg, #0f2d52 0%, #1d4e89 55%, #2e75b6 100%);
        border-radius: 14px; padding: 16px 24px; margin-bottom: 10px;
        display: flex; align-items: center; justify-content: space-between;
        box-shadow: 0 8px 24px rgba(15, 45, 82, 0.25);
    }}
    .hero-title {{ color: white; font-size: 26px; font-weight: 800; margin: 0; }}
    .hero-sub {{ color: #cfe0f5; font-size: 14px; margin-top: 4px; }}
    .live-pill {{
        display: flex; align-items: center; gap: 8px; background: rgba(255,255,255,0.12);
        padding: 8px 16px; border-radius: 999px; color: white; font-weight: 600; font-size: 13px;
    }}
    .live-dot {{ width: 10px; height: 10px; border-radius: 50%; background: #ff5c5c; }}
    .live-dot.running {{ animation: pulse 1.1s infinite; }}
    .live-dot.paused {{ background: #9aa7b5; }}
    @keyframes pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(255,92,92,0.6); }}
        70% {{ box-shadow: 0 0 0 10px rgba(255,92,92,0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(255,92,92,0); }}
    }}

    .kpi-card {{
        background: {card_bg}; border-radius: 14px; padding: 16px 18px; border: 1px solid {card_border};
        box-shadow: 0 2px 8px rgba(16,24,40,0.08); transition: transform 0.15s ease;
    }}
    .kpi-card:hover {{ transform: translateY(-2px); }}
    .kpi-label {{ font-size: 12.5px; color: {label_color}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }}
    .kpi-value {{ font-size: 28px; font-weight: 800; color: {value_color}; margin-top: 2px; }}
    .kpi-sub {{ font-size: 12px; color: {sub_color}; margin-top: 2px; }}

    .alert-fraud {{
        background: linear-gradient(90deg, #c00000, #e23636); color: white; padding: 12px 18px;
        border-radius: 12px; font-weight: 700; font-size: 15px; margin-bottom: 10px;
        animation: flash 0.9s ease-in-out 2;
    }}
    @keyframes flash {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.55; }} }}

    .feed-table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; color: {app_text}; }}
    .feed-table th {{ text-align: left; color: {table_head}; font-size: 11.5px; text-transform: uppercase;
        letter-spacing: 0.03em; padding: 6px 10px; border-bottom: 2px solid {table_border}; }}
    .feed-table td {{ padding: 7px 10px; border-bottom: 1px solid {table_border}; }}
    .feed-table tr:hover {{ background: {row_hover}; }}
    .feed-table tr.latest {{ animation: rowglow 1.4s ease-out 1; }}
    @keyframes rowglow {{ 0% {{ background: {glow_from}; }} 100% {{ background: {glow_to}; }} }}
    .badge {{ padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; }}
    .badge-safe {{ background: {badge_safe_bg}; color: {badge_safe_text}; }}
    .badge-warn {{ background: {badge_warn_bg}; color: {badge_warn_text}; }}
    .badge-danger {{ background: {badge_danger_bg}; color: {badge_danger_text}; }}

    /* Streamlit dat inline style / class rieng cho nut theo tung trang thai (binh thuong, hover,
       focus sau khi bam, disabled luc dang chay mo phong) nen phai ghi de bang !important cho
       TAT CA cac trang thai, neu khong no se roi ve mau trang mac dinh gay vo theme toi/sang */
    section[data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {{
        background-color: {btn_bg} !important; color: {btn_text} !important; border: 1px solid {btn_border} !important;
    }}
    section[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {{
        border-color: {btn_hover_border} !important; color: {btn_hover_border} !important;
    }}
    section[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):focus,
    section[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):focus-visible,
    section[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):active {{
        background-color: {btn_bg} !important; color: {btn_text} !important;
        border-color: {btn_hover_border} !important; box-shadow: none !important;
    }}
    section[data-testid="stSidebar"] .stButton > button:disabled,
    section[data-testid="stSidebar"] .stButton > button[disabled] {{
        background-color: {btn_bg} !important; color: {btn_text} !important;
        border: 1px solid {btn_border} !important; opacity: 0.55 !important;
    }}
    section[data-testid="stSidebar"] .stButton > button[kind="primary"]:disabled,
    section[data-testid="stSidebar"] .stButton > button[kind="primary"][disabled] {{
        opacity: 0.6 !important;
    }}
    /* O chon (selectbox) - Streamlit dung react-aria ComboBox, khung ngoai (role=group) giu
       nen mac dinh sang mau, trong khi chu input bi ep mau theo theme, neu khong set lai nen
       o day thi chu trang se bi chim tren nen trang (hoac nguoc lai) */
    .stApp [data-testid="stSelectbox"] [role="group"],
    .stApp [data-baseweb="select"] > div {{
        background-color: {card_bg} !important; border-color: {card_border} !important;
    }}
    .stApp [data-testid="stSelectbox"] input,
    .stApp [data-baseweb="select"] * {{
        color: {app_text} !important; background-color: transparent !important;
    }}
    .stApp [data-testid="stSelectbox"] svg {{ fill: {app_text} !important; }}

    /* The nho dung chung cho cac khoi o dang luoi (gauge rong khi chua chay, bang trong, ...) */
    .mini-card {{
        background: {card_bg}; border: 1px solid {card_border}; border-radius: 12px;
        padding: 12px 14px; height: 100%;
    }}
    .mini-card-title {{ font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;
        color: {label_color}; margin-bottom: 8px; }}
    .placeholder-box {{
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        height: 236px; border: 1.5px dashed {card_border}; border-radius: 10px;
        color: {sub_color}; font-size: 13px; text-align: center; padding: 0 16px; gap: 6px;
    }}
    .placeholder-box .big-icon {{ font-size: 26px; opacity: 0.6; }}

    .side-info-card {{
        background: {card_bg}; border: 1px solid {card_border}; border-radius: 12px; padding: 10px 14px;
    }}
    .side-info-row {{
        display: flex; justify-content: space-between; align-items: center; font-size: 12.5px;
        padding: 5px 0; border-bottom: 1px solid {table_border};
    }}
    .side-info-row:last-child {{ border-bottom: none; }}
    .side-info-row span {{ color: {label_color}; }}
    .side-info-row b {{ color: {value_color}; font-size: 13px; }}
    .legend-row {{ display: flex; align-items: center; gap: 8px; font-size: 12.5px; padding: 3px 0; }}
    </style>
    """, unsafe_allow_html=True)


def style_fig(fig, dark: bool):
    """Ap dung theme Plotly dong bo voi che do sang/toi cua dashboard (font Times New Roman,
    chu den thuan o che do sang, chu trang thuan o che do toi)."""
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#141c2e" if dark else "#ffffff",
        font=dict(family="Times New Roman, Times, serif", color="#ffffff" if dark else "#000000"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ----------------------------------------------------------------------
# Tai du lieu va model (cache de khong lam lai moi lan rerun)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Đang tải model đã huấn luyện...")
def load_models():
    manifest_path = os.path.join(MODEL_DIR, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    models = {name: joblib.load(os.path.join(MODEL_DIR, fname)) for name, fname in MODEL_FILES.items()}
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    return models, scaler, manifest


@st.cache_data(show_spinner="Đang nạp và tái tạo đúng tập test đã giữ lại lúc huấn luyện...")
def load_test_split(feature_order):
    df = pd.read_csv(CSV_PATH)
    df = df.drop_duplicates()
    X = df.drop(columns=["Class"])[feature_order]
    y = df["Class"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return X_test.reset_index(drop=True), y_test


@st.cache_data(show_spinner="Đang tính xác suất dự đoán của tất cả mô hình trên tập test...")
def compute_all_proba(_models, _scaler, X_test, feature_order):
    X_raw = X_test[feature_order].values
    X_scaled = _scaler.transform(X_raw)
    proba = {}
    proba["Logistic Regression"] = _models["Logistic Regression"].predict_proba(X_scaled)[:, 1]
    proba["Random Forest"] = _models["Random Forest"].predict_proba(X_raw)[:, 1]
    proba["XGBoost"] = _models["XGBoost"].predict_proba(X_raw)[:, 1]
    proba["LightGBM"] = _models["LightGBM"].predict_proba(X_raw)[:, 1]
    proba["MLP + SMOTE"] = _models["MLP + SMOTE"].predict_proba(X_scaled)[:, 1]
    proba["Voting Ensemble"] = np.mean([proba[n] for n in BASE_MODEL_ORDER], axis=0)
    return proba


def evaluate_all(proba, y_test, threshold=0.5):
    rows, roc_data, pr_data, cms = [], {}, {}, {}
    for name in ALL_MODEL_ORDER:
        p = proba[name]
        y_pred = (p >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, p)
        prec_c, rec_c, _ = precision_recall_curve(y_test, p)
        rows.append({
            "Phương pháp": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, p),
            "PR-AUC (AP)": average_precision_score(y_test, p),
        })
        cms[name] = cm
        roc_data[name] = (fpr, tpr)
        pr_data[name] = (rec_c, prec_c)
    return pd.DataFrame(rows).set_index("Phương pháp").loc[ALL_MODEL_ORDER], cms, roc_data, pr_data


# ----------------------------------------------------------------------
# Khoi tao session state
# ----------------------------------------------------------------------
def init_state(n_test, fraud_idx, legit_idx):
    ss = st.session_state
    ss.setdefault("running", False)
    ss.setdefault("tick", 0)
    ss.setdefault("history", [])
    ss.setdefault("cm_running", {"TP": 0, "FP": 0, "TN": 0, "FN": 0})
    if "fraud_order" not in ss:
        ss.fraud_order = np.random.permutation(fraud_idx).tolist()
        ss.fraud_ptr = 0
        ss.legit_order = np.random.permutation(legit_idx).tolist()
        ss.legit_ptr = 0


def draw_next_index(fraud_ratio):
    ss = st.session_state
    if np.random.rand() < fraud_ratio:
        if ss.fraud_ptr >= len(ss.fraud_order):
            ss.fraud_order = np.random.permutation(ss.fraud_order).tolist()
            ss.fraud_ptr = 0
        idx = ss.fraud_order[ss.fraud_ptr]
        ss.fraud_ptr += 1
    else:
        if ss.legit_ptr >= len(ss.legit_order):
            ss.legit_order = np.random.permutation(ss.legit_order).tolist()
            ss.legit_ptr = 0
        idx = ss.legit_order[ss.legit_ptr]
        ss.legit_ptr += 1
    return idx


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    st.session_state.setdefault("dark_mode", False)
    render_css(st.session_state.dark_mode)
    dark = st.session_state.dark_mode

    models, scaler, manifest = load_models()
    feature_order = manifest["feature_order"]
    X_test, y_test = load_test_split(feature_order)
    proba_all = compute_all_proba(models, scaler, X_test, feature_order)

    fraud_idx = np.where(y_test == 1)[0]
    legit_idx = np.where(y_test == 0)[0]
    init_state(len(y_test), fraud_idx, legit_idx)

    # ---------------- Ten de tai / du an ----------------
    st.markdown(f"""
    <div class="project-title"><span class="proj-badge">Đề tài</span>
        <span>Xây dựng hệ thống Big Data phát hiện gian lận thẻ tín dụng<br>theo thời gian thực dựa trên mô hình AI</span>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- Banner ----------------
    dot_class = "running" if st.session_state.running else "paused"
    status_text = "ĐANG CHẠY" if st.session_state.running else "TẠM DỪNG"
    st.markdown(f"""
    <div class="hero-banner">
        <div>
            <p class="hero-title">Giám sát gian lận thẻ tín dụng theo thời gian (mô phỏng)</p>
            <p class="hero-sub">Mô phỏng phát lại {len(y_test):,} giao dịch giữ riêng làm tập test, mô hình chưa từng huấn luyện trên các giao dịch này</p>
        </div>
        <div class="live-pill"><span class="live-dot {dot_class}"></span>{status_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.toggle("🌙 Chế độ tối", key="dark_mode")
        st.divider()
        st.markdown("### Cấu hình mô phỏng")
        model_choice = st.selectbox("Mô hình dùng để chấm điểm", ALL_MODEL_ORDER, index=ALL_MODEL_ORDER.index("XGBoost"))
        threshold = st.slider("Ngưỡng quyết định gian lận", 0.05, 0.95, 0.50, 0.01)
        speed = st.slider("Tốc độ phát (giây / giao dịch)", 0.1, 3.0, 0.7, 0.1)
        fraud_ratio = st.slider("Tỷ lệ gian lận giả lập trong luồng demo (%)", 0, 50, 8, 1) / 100
        st.caption("Tỷ lệ gian lận thật trong dữ liệu gốc chỉ khoảng 0.17%. Thanh trượt này CHỈ chỉnh tần suất "
                   "xuất hiện trong luồng demo để dễ quan sát, không làm thay đổi dự đoán của model.")

        c1, c2 = st.columns(2)
        if c1.button("▶ Bắt đầu", use_container_width=True, type="primary"):
            st.session_state.running = True
        if c2.button("⏸ Tạm dừng", use_container_width=True):
            st.session_state.running = False
        if st.button("↺ Reset phiên mô phỏng", use_container_width=True):
            st.session_state.running = False
            st.session_state.tick = 0
            st.session_state.history = []
            st.session_state.cm_running = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

        st.markdown("---")
        st.markdown("##### Thông tin tập dữ liệu")
        st.markdown(f"""<div class="side-info-card">
            <div class="side-info-row"><span>Tổng giao dịch tập test</span><b>{len(y_test):,}</b></div>
            <div class="side-info-row"><span>Gian lận thật trong tập test</span><b>{int(y_test.sum()):,}</b></div>
            <div class="side-info-row"><span>Tỷ lệ gian lận thật</span><b>{y_test.mean() * 100:.3f}%</b></div>
            <div class="side-info-row"><span>Số đặc trưng đầu vào</span><b>{len(feature_order)}</b></div>
        </div>""", unsafe_allow_html=True)

        if st.button("📁 Chèn đường dẫn dữ liệu", use_container_width=True):
            st.session_state.show_data_path = not st.session_state.get("show_data_path", False)
        if st.session_state.get("show_data_path", False):
            st.code(CSV_PATH, language="text")

        with st.expander("Chú giải nhãn kết quả"):
            st.markdown("""
            <div class="legend-row"><span class="badge badge-danger">Gian lận · Đúng</span> mô hình gắn cờ, đúng là gian lận thật</div>
            <div class="legend-row"><span class="badge badge-warn">Gian lận · Báo nhầm</span> mô hình gắn cờ nhưng thực tế hợp lệ</div>
            <div class="legend-row"><span class="badge badge-warn">Bỏ sót gian lận</span> thực tế gian lận nhưng mô hình bỏ sót</div>
            <div class="legend-row"><span class="badge badge-safe">Hợp lệ</span> hợp lệ, mô hình dự đoán đúng</div>
            """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Giám sát trực tuyến (mô phỏng)", "So sánh mô hình (đánh giá offline)"])

    # ======================================================================
    # TAB 1: Live simulation
    # ======================================================================
    with tab1:
        cm = st.session_state.cm_running
        total = cm["TP"] + cm["FP"] + cm["TN"] + cm["FN"]
        precision = cm["TP"] / (cm["TP"] + cm["FP"]) if (cm["TP"] + cm["FP"]) > 0 else None
        recall = cm["TP"] / (cm["TP"] + cm["FN"]) if (cm["TP"] + cm["FN"]) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) > 0 else None

        kpi_cols = st.columns(5)
        kpi_defs = [
            ("Giao dịch đã xử lý", f"{total:,}", "Trong phiên mô phỏng hiện tại"),
            ("Gian lận thực tế xuất hiện", f"{cm['TP'] + cm['FN']:,}", "Theo nhãn thật của tập test"),
            ("Mô hình gắn cờ gian lận", f"{cm['TP'] + cm['FP']:,}", f"Mô hình: {model_choice}"),
            ("Precision hiện tại", f"{precision:.3f}" if precision is not None else "—", "TP / (TP + FP)"),
            ("Recall hiện tại", f"{recall:.3f}" if recall is not None else "—", "TP / (TP + FN)"),
        ]
        for col, (label, value, sub) in zip(kpi_cols, kpi_defs):
            col.markdown(f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>""", unsafe_allow_html=True)

        history = st.session_state.history
        latest = history[-1] if history else None

        if latest is not None and latest["pred"] == 1:
            st.markdown(f'<div class="alert-fraud">⚠ Mô hình gắn cờ GIAN LẬN · Giao dịch #{latest["tick"]} · '
                        f'Số tiền ${latest["amount"]:,.2f} · Xác suất {latest["proba"]:.1%}</div>',
                        unsafe_allow_html=True)

        col_gauge, col_roll, col_donut = st.columns([1, 1.5, 1], gap="small")

        with col_gauge:
            st.markdown('<div class="section-label">Điểm rủi ro giao dịch gần nhất</div>', unsafe_allow_html=True)
            if latest is not None:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=latest["proba"] * 100,
                    number={"suffix": "%", "font": {"size": 32}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#0f2d52"},
                        "steps": [
                            {"range": [0, 30], "color": "#e4f3e9"},
                            {"range": [30, 70], "color": "#fff3cd"},
                            {"range": [70, 100], "color": "#fde2e2"},
                        ],
                        "threshold": {"line": {"color": "#c00000", "width": 3}, "value": threshold * 100},
                    },
                ))
                fig_gauge.update_layout(height=236, margin=dict(l=16, r=16, t=6, b=6))
                st.plotly_chart(style_fig(fig_gauge, dark), use_container_width=True)
            else:
                st.markdown('<div class="placeholder-box"><span class="big-icon">⏱</span>'
                             'Nhấn "Bắt đầu" ở thanh bên trái<br>để xem điểm rủi ro trực tiếp</div>',
                             unsafe_allow_html=True)

        with col_roll:
            st.markdown('<div class="section-label">Tỷ lệ gian lận trong luồng gần đây (cửa sổ trượt)</div>', unsafe_allow_html=True)
            if history:
                window = history[-ROLLING_WINDOW:]
                df_hist = pd.DataFrame(window)
                df_hist["rolling_fraud_rate"] = df_hist["true_label"].rolling(10, min_periods=1).mean() * 100
                fig_roll = px.line(df_hist, x="tick", y="rolling_fraud_rate", markers=False)
                fig_roll.update_traces(line_color=COLOR_FRAUD, line_width=3)
                fig_roll.update_layout(
                    height=236, margin=dict(l=16, r=16, t=6, b=6),
                    yaxis_title="Tỷ lệ gian lận (%) / 10 g.dịch gần nhất",
                    xaxis_title="Thứ tự giao dịch trong luồng demo",
                )
                st.plotly_chart(style_fig(fig_roll, dark), use_container_width=True)
            else:
                st.markdown('<div class="placeholder-box"><span class="big-icon">📈</span>'
                             'Biểu đồ tỷ lệ gian lận theo thời gian<br>sẽ hiển thị khi mô phỏng chạy</div>',
                             unsafe_allow_html=True)

        with col_donut:
            st.markdown('<div class="section-label">Thành phần tập test (thực tế)</div>', unsafe_allow_html=True)
            fig_donut = go.Figure(go.Pie(
                labels=["Hợp lệ", "Gian lận"],
                values=[int((y_test == 0).sum()), int((y_test == 1).sum())],
                hole=0.58, marker=dict(colors=[COLOR_LEGIT, COLOR_FRAUD]),
                textinfo="percent", textfont=dict(size=12),
            ))
            fig_donut.update_layout(
                height=236, margin=dict(l=10, r=10, t=6, b=6), showlegend=True,
                legend=dict(orientation="h", y=-0.12),
                annotations=[dict(text=f"{len(y_test):,}<br>giao dịch", x=0.5, y=0.5, font_size=12, showarrow=False)],
            )
            st.plotly_chart(style_fig(fig_donut, dark), use_container_width=True)

        st.markdown('<div class="section-label">Luồng giao dịch gần nhất</div>', unsafe_allow_html=True)
        if history:
            rows_html = ""
            for row in reversed(history[-HISTORY_DISPLAY_ROWS:]):
                if row["pred"] == 1 and row["true_label"] == 1:
                    badge = '<span class="badge badge-danger">Gian lận · Đúng</span>'
                elif row["pred"] == 1 and row["true_label"] == 0:
                    badge = '<span class="badge badge-warn">Gian lận · Báo nhầm</span>'
                elif row["pred"] == 0 and row["true_label"] == 1:
                    badge = '<span class="badge badge-warn">Bỏ sót gian lận</span>'
                else:
                    badge = '<span class="badge badge-safe">Hợp lệ</span>'
                row_class = "latest" if row["tick"] == history[-1]["tick"] else ""
                rows_html += (f'<tr class="{row_class}"><td>#{row["tick"]}</td><td>${row["amount"]:,.2f}</td>'
                              f'<td>{row["proba"]:.1%}</td><td>{badge}</td></tr>')
            st.markdown(f"""
            <table class="feed-table">
                <thead><tr><th>Giao dịch</th><th>Số tiền</th><th>Xác suất gian lận</th><th>Kết quả</th></tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="placeholder-box" style="height:120px;"><span class="big-icon">🧾</span>'
                        'Chưa có giao dịch nào được xử lý</div>', unsafe_allow_html=True)

        # --- tick: xu ly MOT giao dich moi roi tu lam moi trang ---
        if st.session_state.running:
            idx = draw_next_index(fraud_ratio)
            p = float(proba_all[model_choice][idx])
            pred = int(p >= threshold)
            true_label = int(y_test[idx])
            st.session_state.tick += 1
            st.session_state.history.append({
                "tick": st.session_state.tick,
                "amount": float(X_test.iloc[idx]["Amount"]),
                "proba": p,
                "pred": pred,
                "true_label": true_label,
            })
            if len(st.session_state.history) > 500:
                st.session_state.history = st.session_state.history[-500:]
            if pred == 1 and true_label == 1:
                cm["TP"] += 1
            elif pred == 1 and true_label == 0:
                cm["FP"] += 1
            elif pred == 0 and true_label == 0:
                cm["TN"] += 1
            else:
                cm["FN"] += 1
            time.sleep(speed)
            st.rerun()

    # ======================================================================
    # TAB 2: So sanh mo hinh (offline, tren toan bo tap test)
    # ======================================================================
    with tab2:
        results_df, cms, roc_data, pr_data = evaluate_all(proba_all, y_test, threshold=0.5)
        st.markdown('<div class="section-label">Bảng so sánh (ngưỡng 0.5, đánh giá trên toàn bộ tập test)</div>', unsafe_allow_html=True)
        st.dataframe(results_df.style.format("{:.4f}").background_gradient(cmap="Blues", subset=["F1-score", "ROC-AUC"]),
                     use_container_width=True, height=246)

        CHART_H = 320
        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1:
            fig_roc = go.Figure()
            for name in ALL_MODEL_ORDER:
                fpr, tpr = roc_data[name]
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name,
                                              line=dict(color=MODEL_COLORS[name],
                                                        width=3 if name == "Voting Ensemble" else 2,
                                                        dash="dash" if name == "Voting Ensemble" else "solid")))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ngẫu nhiên",
                                          line=dict(color="gray", dash="dot")))
            fig_roc.update_layout(title="Đường ROC", xaxis_title="FPR", yaxis_title="TPR", height=CHART_H,
                                   margin=dict(l=10, r=10, t=36, b=10),
                                   legend=dict(orientation="h", y=-0.28, font=dict(size=9)))
            st.plotly_chart(style_fig(fig_roc, dark), use_container_width=True)

        with c2:
            fig_pr = go.Figure()
            for name in ALL_MODEL_ORDER:
                rec_c, prec_c = pr_data[name]
                fig_pr.add_trace(go.Scatter(x=rec_c, y=prec_c, mode="lines", name=name,
                                             line=dict(color=MODEL_COLORS[name],
                                                       width=3 if name == "Voting Ensemble" else 2,
                                                       dash="dash" if name == "Voting Ensemble" else "solid")))
            fig_pr.update_layout(title="Precision-Recall", xaxis_title="Recall", yaxis_title="Precision", height=CHART_H,
                                  margin=dict(l=10, r=10, t=36, b=10),
                                  legend=dict(orientation="h", y=-0.28, font=dict(size=9)))
            st.plotly_chart(style_fig(fig_pr, dark), use_container_width=True)

        with c3:
            metrics_plot = ["Precision", "Recall", "F1-score", "ROC-AUC"]
            fig_bar = go.Figure()
            for metric in metrics_plot:
                fig_bar.add_trace(go.Bar(name=metric, x=ALL_MODEL_ORDER, y=results_df[metric]))
            fig_bar.update_layout(barmode="group", title="So sánh chỉ số", height=CHART_H,
                                   yaxis_title="Giá trị", margin=dict(l=10, r=10, t=36, b=10),
                                   xaxis=dict(tickfont=dict(size=8)),
                                   legend=dict(orientation="h", y=-0.28, font=dict(size=9)))
            st.plotly_chart(style_fig(fig_bar, dark), use_container_width=True)

        with c4:
            cm_model = st.selectbox("Confusion matrix của", ALL_MODEL_ORDER, index=ALL_MODEL_ORDER.index("XGBoost"),
                                     key="cm_select", label_visibility="collapsed")
            cm_sel = cms[cm_model]
            cm_pct = cm_sel / cm_sel.sum(axis=1, keepdims=True) * 100
            text = [[f"{cm_sel[i, j]:,}<br>({cm_pct[i, j]:.2f}%)" for j in range(2)] for i in range(2)]
            fig_cm = go.Figure(go.Heatmap(
                z=cm_sel, x=["Hợp lệ", "Gian lận"], y=["Hợp lệ", "Gian lận"],
                text=text, texttemplate="%{text}", textfont=dict(size=10), colorscale="Blues", showscale=False,
            ))
            fig_cm.update_layout(title=f"Confusion Matrix · {cm_model}", height=CHART_H - 28,
                                  margin=dict(l=10, r=10, t=36, b=10),
                                  xaxis_title="Dự đoán", yaxis_title="Thực tế")
            st.plotly_chart(style_fig(fig_cm, dark), use_container_width=True)

        c5, c6 = st.columns([1.3, 1], gap="small")
        with c5:
            st.markdown('<div class="section-label">Tầm quan trọng đặc trưng (RF / XGBoost / LightGBM)</div>', unsafe_allow_html=True)
            importances = {}
            for name in ["Random Forest", "XGBoost", "LightGBM"]:
                imp = models[name].feature_importances_
                importances[name] = pd.Series(imp / imp.sum(), index=feature_order)
            avg_rank = pd.DataFrame(importances).mean(axis=1).sort_values(ascending=False)
            top_features = avg_rank.head(12).index[::-1]
            fig_imp = go.Figure()
            for name in ["Random Forest", "XGBoost", "LightGBM"]:
                fig_imp.add_trace(go.Bar(name=name, y=top_features, x=importances[name].reindex(top_features),
                                          orientation="h"))
            fig_imp.update_layout(barmode="group", height=360, title="Top 12 đặc trưng quan trọng nhất",
                                  margin=dict(l=10, r=10, t=36, b=10),
                                   xaxis_title="Tầm quan trọng (đã chuẩn hóa)",
                                   legend=dict(orientation="h", y=-0.15, font=dict(size=9)))
            st.plotly_chart(style_fig(fig_imp, dark), use_container_width=True)

        with c6:
            st.markdown('<div class="section-label">t-SNE trên mẫu dữ liệu</div>', unsafe_allow_html=True)
            if "tsne_fig_df" not in st.session_state:
                st.markdown('<div class="placeholder-box" style="height:300px;"><span class="big-icon">🌀</span>'
                            'Bấm nút bên dưới để tính t-SNE<br>(có thể mất 1-2 phút)</div>', unsafe_allow_html=True)
            if st.button("Tính t-SNE trên mẫu dữ liệu", use_container_width=True):
                from sklearn.manifold import TSNE
                with st.spinner("Đang chạy t-SNE..."):
                    rng = np.random.RandomState(RANDOM_STATE)
                    sample_legit = rng.choice(legit_idx, size=min(4000, len(legit_idx)), replace=False)
                    sample_idx = np.concatenate([fraud_idx, sample_legit])
                    X_scaled_sample = scaler.transform(X_test.iloc[sample_idx][feature_order].values)
                    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE, init="pca", learning_rate="auto")
                    X_2d = tsne.fit_transform(X_scaled_sample)
                    st.session_state.tsne_fig_df = pd.DataFrame({
                        "x": X_2d[:, 0], "y": X_2d[:, 1],
                        "Lớp": np.where(y_test[sample_idx] == 1, "Gian lận", "Hợp lệ"),
                        "Xác suất Ensemble": proba_all["Voting Ensemble"][sample_idx],
                    })
            if "tsne_fig_df" in st.session_state:
                fig_tsne = px.scatter(st.session_state.tsne_fig_df, x="x", y="y", color="Lớp",
                                      color_discrete_map={"Hợp lệ": COLOR_LEGIT, "Gian lận": COLOR_FRAUD},
                                      opacity=0.7, height=330, title="t-SNE theo nhãn thực tế")
                fig_tsne.update_layout(margin=dict(l=10, r=10, t=36, b=10), legend=dict(orientation="h", y=-0.15, font=dict(size=9)))
                st.plotly_chart(style_fig(fig_tsne, dark), use_container_width=True)


if __name__ == "__main__":
    main()
