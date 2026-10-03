import os
import json
import joblib
import numpy as np
from datetime import datetime
from kafka import KafkaConsumer
from pymongo import MongoClient
from preprocess_online import preprocess_online

# ==========================================
# 1. CẤU HÌNH KẾT NỐI MONGODB
# ==========================================
MONGO_URI = "mongodb://root:rootpassword@localhost:27017/"

try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    # Kiểm tra thử kết nối
    mongo_client.admin.command('ping')
    db = mongo_client["fraud_db"]
    collection = db["predictions"]   
    print("Kết nối MongoDB thành công!")
except Exception as e:
    print(f"Lỗi kết nối MongoDB: {e}")
    exit(1)

# ==========================================
# 2. CẤU HÌNH AI & KAFKA
# ==========================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PIPELINE_PATH = os.path.join(SRC_DIR, "notebook", "fraud_detection_pipeline.joblib")

BOOTSTRAP_SERVERS = ['localhost:9092']       # Cổng Kafka Broker
TOPIC_NAME = "creditcardfraud"               # Tên Topic Kafka
GROUP_ID = "fraud_detection_production"      # Nhóm Consumer
OPTIMAL_THRESHOLD = 0.9976                   # Ngưỡng dự phòng (sẽ bị ghi đè bởi pipeline)

# ==========================================
# 3. NẠP MÔ HÌNH VÀO RAM
# ==========================================
print("\nNạp mô hình từ fraud_detection_pipeline.joblib")
if not os.path.exists(MODEL_PIPELINE_PATH):
    raise FileNotFoundError(f"Không tìm thấy file mô hình tại: {MODEL_PIPELINE_PATH}")

artifact = joblib.load(MODEL_PIPELINE_PATH)
model = artifact["model"]
OPTIMAL_THRESHOLD = artifact["optimal_threshold"]
scaler_amount = artifact["scaler_amount"]
scaler_time = artifact["scaler_time"]
print("Tải thành công fraud_detection_pipeline.joblib vào bộ nhớ RAM")

# ==========================================
# 4. KHỞI TẠO KAFKA CONSUMER
# ==========================================
print(f"Đang kết nối tới Kafka Broker tại {BOOTSTRAP_SERVERS}")
consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    group_id=GROUP_ID,
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    auto_offset_reset="latest",
    enable_auto_commit=True
)
print(f"Kết nối thành công. Đang lắng nghe bản tin trên topic {TOPIC_NAME}\n")

# ==========================================
# 5. VÒNG LẶP XỬ LÝ REAL-TIME & LƯU MONGODB
# ==========================================
try:
    for message in consumer:
        raw_msg = message.value
        
        # Tiền xử lý dữ liệu
        formatted_data = preprocess_online(raw_msg, scaler_amount, scaler_time)
        if formatted_data is None:
            continue
            
        # Lấy xác suất gian lận
        fraud_probability = float(model.predict_proba(formatted_data)[0][1])
        
        # Phân loại theo Ngưỡng tối ưu
        is_fraud = 1 if fraud_probability > OPTIMAL_THRESHOLD else 0

        # Ép kiểu dữ liệu Numpy về Python Native (bắt buộc để lưu vào MongoDB không bị lỗi)
        enriched_payload = {}
        for k, v in raw_msg.items():
            if isinstance(v, (np.floating, float)):
                enriched_payload[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                enriched_payload[k] = int(v)
            else:
                enriched_payload[k] = v

        # Bổ sung 3 trường kết quả AI
        enriched_payload.update({
            "fraud_probability": round(fraud_probability, 6),
            "is_fraud": is_fraud,
            "processed_at": datetime.now().isoformat()
        })

        # Ghi bản ghi hoàn chỉnh vào MongoDB
        collection.insert_one(enriched_payload)

        # In Log theo dõi
        tx_time = enriched_payload.get("Time", "N/A")
        tx_amount = enriched_payload.get("Amount", 0.0)
        prob_percentage = fraud_probability * 100
        
        if is_fraud == 1:
            print(f"CẢNH BÁO GIAN LẬN (Time: {tx_time} | Số tiền: ${tx_amount:,.2f} | Xác suất: {prob_percentage:.2f}%) | Xử lý lúc: {enriched_payload['processed_at']}")
        else:
            print(f"HỢP LỆ (Time: {tx_time} | Số tiền: ${tx_amount:,.2f} | Xác suất: {prob_percentage:.2f}%)")

except KeyboardInterrupt:
    print("\nĐã dừng lắng nghe Kafka Consumer theo yêu cầu.")
finally:
    consumer.close()
