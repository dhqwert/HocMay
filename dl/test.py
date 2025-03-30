import os
import cv2
import torch
import numpy as np
import json
import datetime
from collections import Counter
from facenet_pytorch import InceptionResnetV1
import joblib

# Load model nhận diện khuôn mặt
embedder = InceptionResnetV1(pretrained='vggface2').eval()

# Load mô hình SVM và KNN
svm_model = joblib.load("model/svm_model.pkl")
knn_model = joblib.load("model/knn_model.pkl")

# Load bộ nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Biến lưu trữ label của 100 frame gần nhất
frame_labels = []
FRAME_BATCH_SIZE = 50  # Dùng voting trên 100 frame 
CONFIDENCE_THRESHOLD = 0.8  # Ngưỡng tin cậy
batch_count = 0

# Đọc dữ liệu từ file nếu tồn tại
if os.path.exists("attendance_log.json"):
    try:
        with open("attendance_log.json", "r") as f:
            attendance_log = json.load(f)
            if not isinstance(attendance_log, dict):  # Kiểm tra kiểu dữ liệu
                attendance_log = {}
    except (json.JSONDecodeError, TypeError):
        attendance_log = {}
else:
    attendance_log = {}

# Hàm tiền xử lý khuôn mặt
def preprocess_face(face):
    face = cv2.resize(face, (160, 160))  # Resize về 160x160
    face = np.stack((face,)*3, axis=-1)  # Chuyển ảnh xám thành ảnh 3 kênh
    face = face.transpose((2, 0, 1)) / 255.0  # Định dạng (3, 160, 160) và chuẩn hóa về [0,1]
    face_tensor = torch.tensor(face).unsqueeze(0).float()  # Thêm batch dimension
    return face_tensor

# Mở camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[INFO] Không thể mở camera!")
    exit()

print("[INFO] Đang chạy... (Nhấn 'q' để thoát)")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]  # Cắt vùng khuôn mặt
        face_tensor = preprocess_face(face)  # Tiền xử lý ảnh
        
        with torch.no_grad():
            face_embedding = embedder(face_tensor).detach().numpy().flatten().reshape(1, -1)  # Trích xuất đặc trưng
        
        # Dự đoán bằng SVM và KNN
        predicted_label_svm = svm_model.predict(face_embedding)[0]
        predicted_label_knn = knn_model.predict(face_embedding)[0]
        
        # Nếu cả hai mô hình đồng thuận, chọn kết quả đó, ngược lại chọn Unknown
        if predicted_label_svm == predicted_label_knn:
            predicted_label = predicted_label_svm
        else:
            predicted_label = "Unknown"
        
        frame_labels.append(predicted_label)

        # Nếu đủ batch, tiến hành voting
        if len(frame_labels) == FRAME_BATCH_SIZE:
            most_common = Counter(frame_labels).most_common(1)
            most_common_label = most_common[0][0] if most_common else "Unknown"
            display_label = most_common_label

            cv2.putText(frame, f"User ID: {display_label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame_labels.clear()
            batch_count += 1
            print(f"[INFO] Batch {batch_count} - Voted Label: {display_label}")

        # Vẽ bounding box lên ảnh
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Hiển thị khung hình
    cv2.imshow("Nhan dien khuon mat", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
print("[INFO] Kết thúc chương trình.")
