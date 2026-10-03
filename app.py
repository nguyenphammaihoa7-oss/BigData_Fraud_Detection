import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Cấu hình trang
st.set_page_config(
    page_title="SOC Fraud Monitor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Khởi tạo Sesion State
if 'last_df' not in st.session_state:
    st.session_state['last_df'] = pd.DataFrame() 

# Kết nối MONGODB và lấy dữ liệu
@st.cache_data(ttl=1)
def fetch_realtime_data(limit=2000):
    try:
        MONGO_URI = "mongodb://root:rootpassword@localhost:27017/"
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        db = client["fraud_db"]
        collection = db["predictions"]
        
        # Sắp xếp theo thời gian mới nhất
        cursor = collection.find().sort("processed_at", -1).limit(limit)
        data = list(cursor)
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        
        # Đảm bảo các cột chuẩn kiểu dữ liệu
        if 'processed_at' in df.columns:
            df['processed_at'] = pd.to_datetime(df['processed_at'])
            
        return df
    except Exception as e:
        # Thay vì trả về data rỗng, ta quăng lỗi ra để khối try-except bên dưới bắt được
        raise e

# Bảng điều khiển
with st.sidebar:
    st.title("Bảng điều khiển giám sát gian lận")
    st.markdown("---")
    # Đèn báo trạng thái hệ thống
    st.markdown("<div class='status-online'>● KAFKA STREAM: ONLINE</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Cấu hình Real-time")
    refresh_rate = st.slider("Tần số Auto-Refresh (giây)", min_value=1, max_value=10, value=2)
    limit_records = st.select_slider("Khung thời gian", options=[100, 300, 500, 1000, 2000], value=500)
    risk_threshold = st.slider("Ngưỡng cảnh báo (%)" , min_value=0.0, max_value=1.0, value=0.35, step=0.05)

    if st.button("Tải lại thủ công", width="stretch"):
        st.cache_data.clear()
        st.rerun()

st_autorefresh(interval=refresh_rate * 1000, key="datarefresh")

# ==========================================
# KHỐI BẮT LỖI (EXCEPTION HANDLING) BẮT ĐẦU
# ==========================================
try:
    # Xử lý tín hiệu
    df_new = fetch_realtime_data(limit=limit_records)

    # Nếu truy vấn mới thành công và có dữ liệu -> cập nhật bộ nhớ đệm
    if not df_new.empty:
        st.session_state['last_df'] = df_new

    df_raw = st.session_state['last_df']
    
    if df_raw.empty:
        st.info("🟢 Hệ thống luồng đang hoạt động trơn tru. Đang chờ những giao dịch đầu tiên từ Kafka dội về...")
        st.stop()
        
    # Tính toán số liệu tổng quan
    total_tx = len(df_raw)
    fraud_df = df_raw[df_raw["is_fraud"] == 1]
    fraud_count = len(fraud_df)
    fraud_rate = (fraud_count / total_tx * 100) if total_tx > 0 else 0.0
    total_fraud_amount = fraud_df['Amount'].sum() if 'Amount' in fraud_df.columns else 0.0
    last_update = datetime.now().strftime("%H:%M:%S")

    # Header và KPI
    col_header, col_time = st.columns([8, 2])
    with col_header: st.title("Hệ thống giám sát gian lận")
    with col_time: st.caption(f"Cập nhật: {last_update}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng giao dịch", f"{total_tx:,}")
    with col2:
        st.metric("Phát hiện gian lận", f"{fraud_count}", delta=f"{fraud_rate:.2f}% tỷ lệ", delta_color="inverse")
    with col3:
        st.metric("Tỷ lệ cảnh báo", f"{(len(df_raw[df_raw['fraud_probability'] >= risk_threshold]) / total_tx * 100):.2f}%")
    with col4:
        st.metric("Thiệt hại ngăn chặn", f"${total_fraud_amount:,.2f}")

    st.markdown("---")

    # Biểu đồ
    col_chart_left, col_chart_right = st.columns([6, 4])

    # Biểu đồ 1: Trend giao dịch theo thời gian
    with col_chart_left:
        st.subheader("Xu hướng giao dịch theo thời gian")
        df_trend = df_raw.copy()
        df_trend.set_index('processed_at', inplace=True)
        df_resampled = df_trend.resample('10s').agg({
            'is_fraud': 'sum',
            'Amount': 'count'
        }).rename(columns={'is_fraud': 'Gian_Lận', 'Amount': 'Tổng_Giao_Dịch'}).reset_index()

        fig_line = px.line(
            df_resampled, 
            x='processed_at', 
            y=['Tổng_Giao_Dịch', 'Gian_Lận'],
            color_discrete_map={'Tổng_Giao_Dịch': '#3498DB', 'Gian_Lận': '#FF4B4B'},
            markers=True,
            template="plotly_dark"
        )
        fig_line.update_xaxes(title_text="Thời gian quét", showgrid=True)
        fig_line.update_yaxes(title_text="Số lượng giao dịch", showgrid=True)
        fig_line.update_layout(
            height=350, 
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            uirevision='constant'
        )
        st.plotly_chart(fig_line, width="stretch", theme="streamlit")

    # Biểu đồ 2: Phân bố tỷ lệ gian lận và hợp lệ 
    with col_chart_right:
        st.subheader("Tỷ lệ phân loại")
        
        labels = ['Hợp lệ (Legit)', 'Gian lận (Fraud)']
        values = [total_tx - fraud_count, fraud_count]
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=.5,
            marker_colors=['#27AE60', '#FF4B4B']
        )])
        fig_donut.update_layout(
            template="plotly_dark",
            height=350, 
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=True,
            uirevision='constant'
        )
        st.plotly_chart(fig_donut, width="stretch")

    # Bảng cảnh báo
    st.markdown("---")
    st.subheader("Bảng cảnh báo giao dịch đáng ngờ")

    # Lọc chỉ lấy các bản ghi nghi vấn hoặc theo ngưỡng threshold từ Sidebar
    display_cols = ['processed_at', 'Amount', 'fraud_probability', 'is_fraud']
    existing_cols = [c for c in display_cols if c in df_raw.columns]
    alerts_df = df_raw[df_raw['fraud_probability'] >= risk_threshold].copy()

    if not alerts_df.empty:
        cols_to_show = {'processed_at': 'Thời gian xử lý', 'Amount': 'Số tiền ($)', 'fraud_probability': 'Mức độ rủi ro', 'is_fraud': 'Đánh giá'}
        available_cols = {k: v for k, v in cols_to_show.items() if k in alerts_df.columns}
        display_df = alerts_df[list(available_cols.keys())].rename(columns=available_cols).sort_values(by=['Mức độ rủi ro', 'Thời gian xử lý'], ascending=[False, False])
            
        display_df['Thời gian xử lý'] = display_df['Thời gian xử lý'].dt.strftime('%H:%M:%S')
        display_df['Số tiền ($)'] = display_df['Số tiền ($)'].apply(lambda x: f"${x:,.2f}")
        display_df['Mức độ rủi ro'] = display_df['Mức độ rủi ro'].apply(lambda x: f"{x*100:.2f}%")
        display_df['Đánh giá'] = display_df['Đánh giá'].apply(lambda x: "GIAN LẬN" if x == 1 else "NGHI VẤN")

        def style_risk_rows(row):
            if "GIAN LẬN" in row['Đánh giá']: return ['background-color: rgba(231, 76, 60, 0.2); color: #FF4B4B; font-weight: bold;'] * len(row)
            return ['color: #F39C12; font-weight: 500;'] * len(row)

        st.dataframe(display_df.style.apply(style_risk_rows, axis=1), width="stretch", height=400, hide_index=False)
    else:
        st.success("Không có giao dịch nào vượt ngưỡng rủi ro trong khoảng thời gian này.")

except Exception as e:
    # Bắt lỗi khi mất kết nối MongoDB hoặc hệ thống sập
    st.error("🔴 CẢNH BÁO: Mất kết nối tới cơ sở dữ liệu hoặc luồng Kafka bị gián đoạn!")
    st.warning(f"Chi tiết mã lỗi hệ thống: {e}")
    st.stop()