from flask import Flask, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

CSV_FILE = "attendance/attendance_total.csv"
JSON_FILE = "attendance/attendance.json"

def read_csv_to_json():
    """Đọc file CSV, xử lý NaN và chuyển thành danh sách JSON"""
    try:
        if not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0:
            return []  # ✅ Nếu file không tồn tại hoặc rỗng, trả về danh sách trống
        
        df = pd.read_csv(CSV_FILE, dtype={"nhanvien_id": str, "ngay": str, "giovao": str, "giora": str})
        df = df.where(pd.notna(df), None)  # ✅ Thay NaN bằng None (JSON-friendly)
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": f"Lỗi khi đọc file CSV: {str(e)}"}

def save_json(data):
    """Lưu dữ liệu JSON vào file"""
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        pd.DataFrame(data).to_json(f, orient="records", force_ascii=False, indent=4)

@app.route("/api/attendance", methods=["GET"])
def get_attendance():
    """API trả về dữ liệu chấm công dưới dạng JSON"""
    data = read_csv_to_json()
    save_json(data)
    return jsonify(data)

@app.route("/api/attendance", methods=["PUT"])
def update_attendance():
    """API cập nhật dữ liệu chấm công từ C#"""
    try:
        # ✅ Kiểm tra JSON hợp lệ
        if not request.is_json:
            return jsonify({"error": "Dữ liệu đầu vào không phải JSON!"}), 400

        data = request.json
        if not data:
            return jsonify({"error": "Dữ liệu đầu vào không hợp lệ!"}), 400

        if not os.path.exists(CSV_FILE):
            return jsonify({"error": "File CSV không tồn tại!"}), 404

        df = pd.read_csv(CSV_FILE, dtype={"nhanvien_id": str, "ngay": str, "giovao": str, "giora": str})

        updated = False
        new_entries = []

        for item in data:
            nhanvien_id = item.get("nhanvien_id", "").strip()
            ngay = item.get("ngay", "").strip()
            giovao = item.get("giovao")
            giora = item.get("giora")

            if not nhanvien_id or not ngay:
                continue  # ✅ Bỏ qua nếu thiếu ID nhân viên hoặc ngày

            # 🔹 Kiểm tra xem bản ghi đã tồn tại chưa
            mask = (df["nhanvien_id"] == nhanvien_id) & (df["ngay"] == ngay)

            if mask.any():
                # ✅ Cập nhật dữ liệu nếu tồn tại
                df.loc[mask, "giovao"] = giovao if giovao is not None else df.loc[mask, "giovao"].values[0]
                df.loc[mask, "giora"] = giora if giora is not None else df.loc[mask, "giora"].values[0]
            else:
                # ✅ Nếu không tồn tại, thêm vào danh sách cần thêm
                new_entries.append({
                    "nhanvien_id": nhanvien_id,
                    "ngay": ngay,
                    "giovao": giovao if giovao is not None else "",
                    "giora": giora if giora is not None else "",
                })
            updated = True

        # ✅ Thêm các bản ghi mới vào DataFrame
        if new_entries:
            df = pd.concat([df, pd.DataFrame(new_entries)], ignore_index=True)

        if not updated:
            return jsonify({"error": "Không tìm thấy dữ liệu để cập nhật."}), 404

        # ✅ Lưu lại vào CSV
        df.to_csv(CSV_FILE, index=False)

        # ✅ Cập nhật JSON
        updated_data = read_csv_to_json()
        save_json(updated_data)

        return jsonify({"message": "Dữ liệu đã cập nhật thành công!", "data": updated_data})
    
    except Exception as e:
        return jsonify({"error": f"Lỗi khi cập nhật dữ liệu: {str(e)}"}), 500


if __name__ == "__main__":
    os.makedirs("attendance", exist_ok=True)  # ✅ Đảm bảo thư mục tồn tại
    app.run(host="0.0.0.0", port=5001, debug=True)
