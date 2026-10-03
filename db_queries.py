import pandas as pd
from pymongo import MongoClient
import streamlit as st

# Sử dụng cache của Streamlit để tránh tạo lại kết nối DB mỗi khi giao diện refresh
@st.cache_resource
def get_db_collection():
    try:
        MONGO_URI = "mongodb://root:rootpassword@127.0.0.1:27017/admin?authSource=admin"
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        db = client["fraud_db"]
        return db["predictions"]
    except Exception as e:
        st.error(f"Lỗi kết nối cơ sở dữ liệu: {e}")
        return None

def get_summary_metrics(collection):
    """Lấy tổng số giao dịch và tổng số gian lận để hiển thị trên Dashboard"""
    total_transactions = collection.count_documents({})
    total_frauds = collection.count_documents({"is_fraud": 1})
    
    fraud_rate = 0
    if total_transactions > 0:
        fraud_rate = round((total_frauds / total_transactions) * 100, 2)
        
    return total_transactions, total_frauds, fraud_rate

def get_recent_data(collection, limit=100):
    """Lấy danh sách giao dịch gần nhất, chuyển thành Pandas DataFrame"""
    # Lấy dữ liệu và sắp xếp theo thời gian mới nhất (giảm dần)
    cursor = collection.find({}, {"_id": 0}).sort("processed_at", -1).limit(limit)
    df = pd.DataFrame(list(cursor))
    return df

def get_fraud_alerts(collection, limit=20):
    """Lấy danh sách các giao dịch bị đánh dấu là gian lận"""
    cursor = collection.find({"is_fraud": 1}, {"_id": 0}).sort("processed_at", -1).limit(limit)
    df = pd.DataFrame(list(cursor))
    return df