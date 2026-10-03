import sys
from pymongo import MongoClient, errors, ASCENDING, DESCENDING

# Cấu hình chuỗi kết nối tới MongoDB Container
MONGO_URI = "mongodb://root:rootpassword@localhost:27017/"
DB_NAME = "fraud_db"

def get_mongo_client(uri: str = MONGO_URI) -> MongoClient:
    """
    Hàm tạo và kiểm tra kết nối tới MongoDB Client.
    """
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Kiểm tra xem Server MongoDB có đang hoạt động không
        client.admin.command('ping')
        print("--> [1] Kết nối thành công tới MongoDB Server!")
        return client
    except errors.ServerSelectionTimeoutError as err:
        print(f"[X] Lỗi kết nối MongoDB: Vui lòng kiểm tra lại Container Docker! ({err})")
        sys.exit(1)

def init_database():
    """
    Hàm khởi tạo Database, các Collections và thiết lập Indexing.
    """
    client = get_mongo_client()
    db = client[DB_NAME]
    
    # 1. Khởi tạo Collection 'transactions' (Lưu toàn bộ giao dịch từ Kafka)
    transactions_col = db["predictions"]

    # 2. Đánh Index (Chỉ mục) giúp tối ưu tốc độ truy vấn khi dữ liệu lớn (Big Data)
    print("--> [2] Đang thiết lập Indexing cho các Collections...")
    transactions_col.create_index([("processed_at", DESCENDING)])       # Cho sort theo thời gian
    transactions_col.create_index([("is_fraud", ASCENDING)])            # Cho filter gian lận
    transactions_col.create_index([("fraud_probability", DESCENDING)])  # Cho filter theo ngưỡng
    
    print(f"--> [3] Khởi tạo hoàn tất Database '{DB_NAME}'!")
    print("    Danh sách Collections hiện có:", db.list_collection_names())
    
    return db

if __name__ == "__main__":
    init_database()