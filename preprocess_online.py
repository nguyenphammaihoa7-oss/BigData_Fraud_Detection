import os
import joblib
import pandas as pd
import numpy as np
import json

# Định nghĩa danh sách các trường đặc trưng chuẩn theo model
FEATURE_NAMES = [
    'Time', 
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 
    'Amount'
]

# 1. Kiểm tra cấu trúc và tính hợp lệ của message từ Kafka
def validate_raw_message(raw_msg):
    if isinstance(raw_msg, (str, bytes)):
        try:
            data = json.loads(raw_msg)
        except json.JSONDecodeError:
            return None, "Không thể giải mã được bản tin thô"
    elif isinstance(raw_msg, dict):
        data = raw_msg.copy()
    else:
        return None, f"Kiểu dữ liệu bản tin không hợp lệ: {type(raw_msg)}"
    
    if 'Time' not in data:
        data['Time'] = data.get('id', 0)

    # Kiểm tra các trường dữ liệu bắt buộc
    for feature in FEATURE_NAMES:
        if feature not in data:
            return None, f"Bản tin bị thiếu trường: {feature}"

    # Ép kiểu dữ liệu về số
    clean_data = {}
    for feature in FEATURE_NAMES:
        val = data[feature]
        if val is None:
            return None, f"Trường {feature} chứa giá trị rỗng"
        try:
            clean_data[feature] = float(val)
        except (ValueError, TypeError):
            return None, f"Trường {feature} chứa giá trị không phải số: '{val}'"
            
    return clean_data, None 

# 2. Trích xuất và định dạng đặc trưng
def extract_and_structure_features(clean_data):
    clean_data["Time"] = (clean_data["Time"] / 3600) % 24
    feature_value = [clean_data[i] for i in FEATURE_NAMES]
    df_features = pd.DataFrame([feature_value], columns=FEATURE_NAMES)
    return df_features

# 3. Chuẩn hóa đặc trưng (Dùng trực tiếp scaler truyền từ Consumer)
def scale_features(df_features, scaler_amount, scaler_time):
    df_scaled = df_features.copy()
    
    # Nếu scaler tồn tại thì thực hiện transform, ngược lại giữ nguyên
    if scaler_amount is not None:
        df_scaled["Amount"] = scaler_amount.transform(df_scaled[["Amount"]])
    if scaler_time is not None:
        df_scaled["Time"] = scaler_time.transform(df_scaled[["Time"]])
        
    return df_scaled

# 4. Đóng gói xuất numpy array
def format_output_for_inference(df_scaled):
    df_ordered = df_scaled[FEATURE_NAMES]
    output_array = df_ordered.to_numpy()
    return output_array

# 5. Main pipeline function
def preprocess_online(raw_msg, scaler_amount, scaler_time):
    clean_data, error = validate_raw_message(raw_msg)
    if error:
        print(f"Cảnh báo: {error}")
        return None

    df_features = extract_and_structure_features(clean_data)

    # Sử dụng bộ scaler được truyền trực tiếp từ Consumer (không gọi load_scalers nữa)
    df_scaled = scale_features(df_features, scaler_amount, scaler_time)

    final_input = format_output_for_inference(df_scaled)

    return final_input