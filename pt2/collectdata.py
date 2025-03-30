import cv2
import os
import sqlite3
import time
import numpy as np

# Kết nối database SQLite
conn = sqlite3.connect("employee.db")
cursor = conn.cursor()

# Tạo bảng nếu chưa có
cursor.execute("""
    CREATE TABLE IF NOT EXISTS Employee (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL
    )
""")
conn.commit()

# Nhập ID và tên nhân viên
user_id = input("Nhập ID nhân viên: ")
name = input("Nhập tên nhân viên: ")

# Lưu vào database
cursor.execute("INSERT OR REPLACE INTO Employee (id, name) VALUES (?, ?)", (user_id, name))
conn.commit()

# Tạo thư mục lưu ảnh
dataset_path = f"dataset_org/{user_id}"
os.makedirs(dataset_path, exist_ok=True)

# Khởi tạo camera và bộ nhận diện khuôn mặt
cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

count = 0  # Số ảnh thu thập
while count < 20: # Thu thập 20 ảnh
    ret, frame = cap.read()
    if not ret:
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]  # Cắt ảnh khuôn mặt
        face = cv2.resize(face, (100, 100))
        
        # Lưu ảnh gốc
        cv2.imwrite(f"{dataset_path}/face_{count+1}.jpg", face)
        count += 1
        
        # Hiển thị bounding box nhưng không lưu
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        time.sleep(1)  # Đợi 0.1s giữa các lần chụp
    cv2.imshow("Thu thập dữ liệu", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Hoàn thành thu thập dữ liệu (phương thứ 2)! Tổng số ảnh: 20")
cap.release()
cv2.destroyAllWindows()
conn.close()
