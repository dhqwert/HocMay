import os
import csv
import cv2
import datetime
import json
import numpy as np
import joblib
from skimage.feature import hog
from collections import Counter
import sqlite3
import re

# Load mô hình PCA và SVM
_, _, _, _, pca = joblib.load("train/preprocessed_data.pkl")
svm = joblib.load("model/svm_hog.pkl")

# Load bộ nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Kết nối database để lấy user_id
conn = sqlite3.connect("employee.db")
cursor = conn.cursor()
# Lấy danh sách ID và tên nhân viên
# cursor.execute("SELECT id, name FROM Employee")
# employee_dict = {str(row[0]): row[1] for row in cursor.fetchall()}  # Lưu dưới dạng {id: name}
# user_ids = [str(row[0]) for row in cursor.fetchall()]  # Lấy danh sách user_id dưới dạng string

# Lấy danh sách ID và tên nhân viên
cursor.execute("SELECT id, name FROM Employee")
data = cursor.fetchall()  # Chỉ gọi fetchall() một lần

# Lưu dưới dạng {id: name}
employee_dict = {str(row[0]): row[1] for row in data}

# Lấy danh sách user_id dưới dạng string
user_ids = list(employee_dict.keys())  # Chỉ cần lấy các key từ employee_dict


# Biến lưu trữ label của 10 frame gần nhất
frame_labels = []
FRAME_BATCH_SIZE = 100  # Dùng voting trên 10 frame 
CONFIDENCE_THRESHOLD = 0.8  # Ngưỡng tin cậy
batch_count = 0

import json
import os

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

def create_daily_csv(attendance_log, employee_list, output_folder="attendance"):    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Kiểm tra ngày hợp lệ
    dates_set = set()
    for emp_id, days in attendance_log.items():
        if isinstance(days, dict):
            for day in days.keys():
                if isinstance(day, str) and re.match(r"\d{4}-\d{2}-\d{2}", day):
                    dates_set.add(day)
                else:
                    print(f"[WARNING] Dữ liệu ngày không hợp lệ: {day}")

    # Ghi file tổng hợp
    csv_total_filename = os.path.join(output_folder, "attendance_total.csv")
    with open(csv_total_filename, mode="w", newline="", encoding="utf-8") as csvfile_total:
        writer_total = csv.writer(csvfile_total)
        writer_total.writerow(["nhanvien_id", "ngay", "giovao", "giora"])

        for day in sorted(dates_set):
            for emp_id in employee_list:                
                if emp_id in attendance_log and day in attendance_log[emp_id]:
                    record = attendance_log[emp_id][day]
                    check_in_str = record.get("check_in", "")
                    check_out_str = record.get("check_out", "")

                    check_in = datetime.datetime.strptime(check_in_str, "%H:%M:%S").time() if check_in_str else None
                    check_out = datetime.datetime.strptime(check_out_str, "%H:%M:%S").time() if check_out_str else None

                    # dimuon = "Y" if check_in and check_in > datetime.time(8, 0, 0) else "N"
                    # vesom = "Y" if check_out and check_out < datetime.time(17, 0, 0) else "N"

                    writer_total.writerow([emp_id, day, check_in_str, check_out_str])
                else:
                    writer_total.writerow([emp_id, day, "", ""])  # Mặc định không đi muộn, không về sớm

    print(f"[INFO] Đã tạo file CSV tổng hợp: {csv_total_filename}")

    # Ghi file theo ngày
    for day in sorted(dates_set):
        d = datetime.datetime.strptime(day, "%Y-%m-%d")
        daily_filename = f"{output_folder}/attendance_{d.day:02}{d.month:02}{d.year}.csv"

        with open(daily_filename, mode="w", newline="", encoding="utf-8") as csvfile_daily:
            writer_daily = csv.writer(csvfile_daily)
            writer_daily.writerow(["nhanvien_id", "ngay", "giovao", "giora"])

            for emp_id in employee_list:
                if emp_id in attendance_log and day in attendance_log[emp_id]:
                    record = attendance_log[emp_id][day]
                    check_in_str = record.get("check_in", "")
                    check_out_str = record.get("check_out", "")

                    check_in = datetime.datetime.strptime(check_in_str, "%H:%M:%S").time() if check_in_str else None
                    check_out = datetime.datetime.strptime(check_out_str, "%H:%M:%S").time() if check_out_str else None

                    # dimuon = "Y" if check_in and check_in > datetime.time(8, 0, 0) else "N"
                    # vesom = "Y" if check_out and check_out < datetime.time(17, 0, 0) else "N"

                    writer_daily.writerow([emp_id, day, check_in_str, check_out_str])
                else:
                    writer_daily.writerow([emp_id, day, "", ""])  # Mặc định không đi muộn, không về sớm

        print(f"[INFO] Đã tạo file CSV theo ngày: {daily_filename}")

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
        face = cv2.resize(face, (100, 100))  # Resize về 100x100

        # Trích xuất đặc trưng HOG
        hog_features = hog(face, pixels_per_cell=(8, 8), cells_per_block=(2, 2)).reshape(1, -1)

        # Giảm chiều bằng PCA
        face_pca = pca.transform(hog_features)

        # Dự đoán với SVM
        predicted_label = str(svm.predict(face_pca)[0])
        confidence = svm.decision_function(face_pca).max()

        # Nếu confidence thấp, gán "Unknown"
        if confidence < CONFIDENCE_THRESHOLD:
            predicted_label = "Unknown"

        emp_id = predicted_label  # Cập nhật ID nhân viên
        frame_labels.append(emp_id)

        # Nếu đủ batch, tiến hành voting
        if len(frame_labels) == FRAME_BATCH_SIZE:
            most_common = Counter(frame_labels).most_common(1)
            if most_common:
                most_common_label, count = most_common[0]
            else:
                most_common_label = "Unknown"

            display_label = most_common_label if most_common_label != "Unknown" else "Unknown"

            # Hiển thị nhãn vote được
            cv2.putText(frame, f"User ID: {display_label} -  Name: {employee_dict[display_label]}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Ghi log chấm công vào file sau mỗi batch
            with open("attendance_log.json", "w") as f:
                json.dump(attendance_log, f, indent=4)

            # Reset buffer để bắt đầu batch mới
            frame_labels.clear()
            batch_count += 1
            print(f"[INFO] Batch {batch_count} - Voted Label: {display_label}")

            emp_id = most_common_label  # Gán nhãn cuối cùng

            # Ghi log chấm công nếu là nhân viên hợp lệ
            if emp_id != "Unknown":
                today = datetime.datetime.now().strftime("%Y-%m-%d")
                current_time = datetime.datetime.now().strftime("%H:%M:%S")

                if emp_id not in attendance_log:
                    attendance_log[emp_id] = {}

                if today not in attendance_log[emp_id]:
                    attendance_log[emp_id][today] = {"check_in": current_time, "check_out": current_time}
                else:
                    attendance_log[emp_id][today]["check_out"] = current_time


                print(f"[INFO] Đã chấm công cho nhân viên {emp_id} - Check-in: {attendance_log[emp_id][today]['check_in']} | Check-out: {attendance_log[emp_id][today]['check_out']}")

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
create_daily_csv(attendance_log, employee_list=user_ids)
conn.close()
print("[INFO] Kết thúc chương trình.")

"""_summary_
    vấn đề svm: gắn nhãn gần nhất cho dù label không có trong db 
"""