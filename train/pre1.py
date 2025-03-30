import os
import cv2
import albumentations as A
from skimage import io

# Thư mục ảnh gốc và ảnh tăng cường
INPUT_DIR = "dataset_org"
OUTPUT_DIR = "dataset_aug"

# Tạo thư mục output nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cấu hình các phép biến đổi tăng cường dữ liệu
augment = A.Compose([
    A.HorizontalFlip(p=0.5),  # Lật ngang
    A.Rotate(limit=10, p=0.5),  # Xoay nhẹ trong khoảng ±10 độ
    A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),  
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),  
    A.RandomGamma(gamma_limit=(90, 110), p=0.3),  
])


# Duyệt qua từng lớp và tăng cường dữ liệu
for class_name in os.listdir(INPUT_DIR):
    input_class_dir = os.path.join(INPUT_DIR, class_name)
    output_class_dir = os.path.join(OUTPUT_DIR, class_name)
    
    if not os.path.isdir(input_class_dir):
        continue
    
    os.makedirs(output_class_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_class_dir) if f.endswith(('.jpg', '.png'))]
    num_images = len(image_files)
    
    # Xác định số lượng ảnh tăng cường dựa trên số lượng ảnh gốc
    if num_images >= 10:
        num_aug = int(num_images * (1/2))  # Tạo thêm 50% số ảnh gốc
    else:
        num_aug = int(num_images * (3/2))  # Tạo thêm 150% số ảnh gốc
    
    print(f"📂 {class_name}: {num_images} ảnh gốc -> Tạo thêm {num_aug} ảnh augment.")
    
    # Lưu ảnh gốc vào thư mục đầu ra
    for img_name in image_files:
        img_path = os.path.join(input_class_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(output_class_dir, img_name), img)
    
    # Tạo ảnh augment từ ảnh gốc
    for i in range(num_aug):
        img_name = image_files[i % num_images]  # Chọn ảnh gốc ngẫu nhiên để augment
        img_path = os.path.join(input_class_dir, img_name)
        img = io.imread(img_path)
        augmented = augment(image=img)['image']
        aug_name = f"aug_{i}_{img_name}"
        io.imsave(os.path.join(output_class_dir, aug_name), augmented)

print("🎉 Hoàn thành tăng cường dữ liệu! Ảnh gốc vẫn được giữ nguyên, ảnh augment được tạo thêm theo tỷ lệ quy định.")
