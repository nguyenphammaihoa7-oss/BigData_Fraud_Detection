import json
import time
import pandas as pd
from confluent_kafka import Producer

# Cấu hình nâng cao cho Kafka Producer (Big Data Standards)
conf = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'advanced-credit-card-producer',
    
    # 1. Đảm bảo tin cậy dữ liệu (Reliability)
    'acks': 'all',                      # Đợi toàn bộ replicas xác nhận để không mất tin nhắn
    'retries': 5,                       # Thử lại tối đa 5 lần nếu gặp lỗi mạng chập chờn
    'retry.backoff.ms': 500,            # Khoảng thời gian chờ giữa các lần thử lại (ms)
    
    # 2. Tối ưu hóa hiệu năng & Băng thông (Performance & Throughput)
    'compression.type': 'snappy',       # Nén dữ liệu bằng thuật toán Snappy
    'linger.ms': 20,                    # Chờ tối đa 20ms để gom tin nhắn thành batch
    'batch.num.messages': 500           # Kích thước tối đa của 1 batch
}

producer = Producer(conf)

def delivery_report(err, msg):
    if err is not None:
        print(f"[X] Gửi thất bại: {err}")
    else:
        print(f"--> Đã gửi giao dịch ID {msg.key().decode('utf-8')} tới Topic {msg.topic()} [Partition {msg.partition()}]")

# Đọc dữ liệu từ file CSV
csv_path = "data/creditcard.csv" 
print("--> Đang đọc file CSV...")
df = pd.read_csv(csv_path)

# Đổi tên topic phát dữ liệu thành 'creditcardfraud' để khớp với Consumer
TARGET_TOPIC = 'creditcardfraud'
print(f"--> Bắt đầu phát dữ liệu vào Kafka Topic '{TARGET_TOPIC}'...")

try:
    for index, row in df.iterrows():
        record = row.to_dict()
        json_payload = json.dumps(record)
        
        producer.produce(
            topic=TARGET_TOPIC,
            key=str(int(record.get('id', index))),
            value=json_payload,
            callback=delivery_report
        )
        
        producer.poll(0)
        time.sleep(0.1) # Phát 10 giao dịch/giây

except KeyboardInterrupt:
    print("\n--> Đã dừng Producer.")

finally:
    producer.flush()