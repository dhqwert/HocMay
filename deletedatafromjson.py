import json

with open("attendance_log.json", "w") as f:
    json.dump({}, f, indent=4)  # Ghi vào file một object JSON rỗng
print("[INFO] Đã xóa toàn bộ dữ liệu chấm công.")
