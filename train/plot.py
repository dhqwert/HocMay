import os
import matplotlib.pyplot as plt

# Hàm đếm số lượng ảnh trong mỗi lớp của một dataset
def count_images_in_dataset(dataset_path):
    class_counts = {}
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):  # Kiểm tra nếu là thư mục
            num_images = len(os.listdir(class_dir))
            class_counts[class_name] = num_images
    return dict(sorted(class_counts.items(), key=lambda x: x[1], reverse=True))  # Sắp xếp giảm dần

# Dataset paths
dataset_paths = {"dataset_org": "dataset_org", "dataset_aug": "dataset_aug"}

# Đếm số ảnh trong từng dataset
dataset_counts = {name: count_images_in_dataset(path) for name, path in dataset_paths.items()}

# Vẽ từng biểu đồ riêng biệt
for dataset_name, class_counts in dataset_counts.items():
    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Lớp (Class)")
    plt.ylabel("Số lượng ảnh")
    plt.title(f"Phân bố số lượng mẫu trong mỗi lớp trong thư mục {dataset_name}")
    plt.tight_layout()  # Tự động điều chỉnh khoảng cách để tránh bị che
    plt.show()
