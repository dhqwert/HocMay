import sqlite3

# Kết nối database
conn = sqlite3.connect("employee.db")
cursor = conn.cursor()

# Danh sách dữ liệu cần chèn
data = [
    ('BCS230008', "Nguyen Tuan Anh"),
    ('BCS230011', "Nguyen Duc Anh"),
    ('BCS230021', "Pham Tien Dat"),
    ('BCS230035', "Lai Hoang Duy"),
    ('BCS230039', "Vu Van Huan"),
    ('BCS230064', "Nguyen Thi Ngoc"),
    ('BCS230073', "Nguyen Ba Minh Quang"),
    ('BCS230085', "Pham Viet Trinh"),
    ('BCS230093', "Nguyen Dinh Quang Vinh"),
    ('BCS230099', "Nguyen Viet Duc"),
    ('BCS230116', "Nguyen Thanh Vinh"),
    ('BCS230119', "Cao Nguyen Minh Quan"),
    ('BCS230123', "Quach Ngoc Nguyen"),
]

# Chèn dữ liệu vào Employee, nếu đã có thì cập nhật
cursor.executemany("INSERT OR REPLACE INTO Employee (id, name) VALUES (?, ?)", data)
# cursor.execute("DELETE FROM Employee WHERE id = 'BCS230008'")
# cursor.execute("DELETE FROM Employee WHERE id = 'BCS230011'")
# cursor.execute("DELETE FROM Employee WHERE id = 'BCS230021'")
# cursor.execute("DELETE FROM Employee WHERE id = 'BCS230035'")
# cursor.execute("DELETE FROM Employee WHERE id = 'BCS230039'")
# cursor.execute("DELETE FROM Employee WHERE id = 'BCS230064'")
# cursor.execute("DELETE FROM Employee WHERE id = 'BCS230073'")
# cursor.execute("DELETE FROM Employee WHERE id = 'BCS230085'")
# cursor.execute("DELETE FROM Employee WHERE id = 'BCS230093'")
# cursor.execute("DELETE FROM Employee WHERE id = 'BCS230099'")
# cursor.execute("DELETE FROM Employee WHERE id = 'BCS230116'")
# cursor.execute("DELETE FROM Employee WHERE id = 'BCS230119'")
# cursor.execute("DELETE FROM Employee WHERE id = 'BCS230123'")

cursor.execute("select * from Employee")

# Lưu thay đổi và đóng kết nối
conn.commit()
conn.close()

print("✅ Đã chèn dữ liệu vào Employee!")
