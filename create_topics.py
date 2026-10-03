from confluent_kafka.admin import AdminClient, NewTopic, NewPartitions

def setup_kafka_topic():
    admin_client = AdminClient({'bootstrap.servers': 'localhost:9092'})
    topic_name = 'transactions'
    target_partitions = 3

    print(f"--> Đang kiểm tra cấu hình Topic '{topic_name}'...")
    
    metadata = admin_client.list_topics(timeout=5)
    existing_topic = metadata.topics.get(topic_name)

    # 1. Nếu Topic chưa tồn tại -> Tạo mới 3 Partitions
    if existing_topic is None or existing_topic.error is not None:
        print(f"--> Khởi tạo mới Topic '{topic_name}' với {target_partitions} Partitions...")
        new_topic = NewTopic(
            topic=topic_name, 
            num_partitions=target_partitions, 
            replication_factor=1,
            config={'retention.ms': '604800000'}
        )
        fs = admin_client.create_topics([new_topic])
        for topic, f in fs.items():
            f.result()
        print(f"--> [Thành công] Đã tạo mới Topic '{topic_name}'!")
    
    # 2. Nếu Topic đã có -> Tăng số Partitions lên 3 (Sửa chuẩn cú pháp List)
    else:
        current_partitions = len(existing_topic.partitions)
        print(f"--> Topic '{topic_name}' hiện tại đang có {current_partitions} Partition(s).")
        
        if current_partitions < target_partitions:
            print(f"--> Đang mở rộng số Partitions từ {current_partitions} lên {target_partitions}...")
            # SỬA LẠI ĐÚNG CÚ PHÁP: Truyền List [NewPartitions(...)] thay vì Dictionary
            new_parts = [NewPartitions(topic_name, target_partitions)]
            fs = admin_client.create_partitions(new_parts)
            for topic, f in fs.items():
                f.result()
            print(f"--> [Thành công] Đã nâng cấp Topic '{topic_name}' lên {target_partitions} Partitions!")
        else:
            print(f"--> [OK] Topic '{topic_name}' đã đạt chuẩn {target_partitions} Partitions.")

if __name__ == "__main__":
    setup_kafka_topic()