from db_connection import init_database, DB_NAME

db = init_database()

# Xóa bỏ hẳn Collection cũ để giải phóng Index unique bị kẹt
db["predictions"].drop()

print("--> Đã xóa hoàn toàn các Collection và Index cũ!")
init_database()