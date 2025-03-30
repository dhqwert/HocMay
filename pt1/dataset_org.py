import cv2
import os
import numpy as np

# Load Haar cascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Thư mục chứa ảnh gốc và ảnh đã xử lý
INPUT_DIR = "dataset_nodb" # ảnh từ dự án trước
OUTPUT_DIR = "dataset_org"

# Đảm bảo thư mục output tồn tại
os.makedirs(OUTPUT_DIR, exist_ok=True)

def align_face(image, face):
    """
    Căn chỉnh khuôn mặt nếu bị nghiêng (dùng mắt, mũi để chuẩn hóa góc).
    """
    # Chuyển ảnh về grayscale nếu chưa có
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    x, y, w, h = face
    face_roi = gray[y:y+h, x:x+w]  # Cắt vùng mặt

    # Resize về 100x100
    aligned_face = cv2.resize(face_roi, (100, 100))

    return aligned_face

def process_images():
    """
    Duyệt toàn bộ dataset, phát hiện khuôn mặt, crop, resize và lưu ảnh mới.
    """
    for user_id in os.listdir(INPUT_DIR):
        user_path = os.path.join(INPUT_DIR, user_id)
        output_user_path = os.path.join(OUTPUT_DIR, user_id)

        if not os.path.isdir(user_path):
            continue

        os.makedirs(output_user_path, exist_ok=True)

        for img_name in os.listdir(user_path):
            img_path = os.path.join(user_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh xám

            if img is None:
                print(f"⚠ Lỗi đọc ảnh: {img_path}")
                continue

            # Phát hiện khuôn mặt
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

            if len(faces) > 0:
                # Chọn khuôn mặt lớn nhất (gần camera nhất)
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                aligned_face = align_face(img, largest_face)

                # Lưu ảnh đã xử lý
                save_path = os.path.join(output_user_path, img_name)
                cv2.imwrite(save_path, aligned_face)
            else:
                print(f"⚠ Không tìm thấy khuôn mặt trong: {img_path}")

    print("✅ Hoàn thành thu thập dữ liệu phương thức 1!")

if __name__ == "__main__":
    process_images()

