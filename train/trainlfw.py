import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import fetch_lfw_people
from skimage.feature import hog
from skimage.transform import resize
import joblib
import time
import cv2
from sklearn.metrics import accuracy_score


# 1️⃣ Tải dữ liệu LFW
print("Đang tải dữ liệu LFW...")
lfw_people = fetch_lfw_people(min_faces_per_person=100, resize=1.0, download_if_missing=True)
X, Y = lfw_people.images, lfw_people.target
print(f"Tổng số ảnh: {len(X)}, Số lớp: {len(set(Y))}")

# 2️⃣ Tiền xử lý: Resize ảnh để đảm bảo cùng kích thước
fixed_size = (50, 37)
X_resized = np.array([resize(img, fixed_size, anti_aliasing=True) for img in X])

def extract_hog_features(images):
    """Trích xuất đặc trưng HOG."""
    features = [hog(img, pixels_per_cell=(4, 4), cells_per_block=(2, 2)) for img in images]
    return np.array(features)

# Trích xuất đặc trưng HOG
print("Đang trích xuất đặc trưng HOG...")
# start_time = time.time()
X_hog = extract_hog_features(X_resized)
print(f"Feature HOG mỗi ảnh: {X_hog.shape[1]}")

# trích xuất đặc trưng LBP
# def extract_lbp_features(images):
#     """Trích xuất đặc trưng LBP từ tập ảnh."""
#     features = []
#     for img in images:
#         gray = (img * 255).astype(np.uint8)  # Chuyển về định dạng uint8
#         lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
#         features.append(lbp.flatten())
#     return np.array(features)

# print("Đang trích xuất đặc trưng LBP...")
# X_lbp = extract_lbp_features(X)
# print(f"Feature LBP mỗi ảnh: {X_lbp.shape[1]}")

# 3️⃣ Chia tập Train/Test
X_train, X_test, Y_train, Y_test = [], [], [], []

for label in np.unique(Y):
    X_class = X_hog[Y == label]
    Y_class = Y[Y == label]

    # Nếu lớp chỉ có 1 mẫu, đưa thẳng vào tập train
    if len(Y_class) == 1:
        X_train.append(X_class)
        Y_train.append(Y_class)
        continue
    
    # Nếu lớp có ít hơn 5 mẫu, giảm test_size xuống 10%
    test_size = 0.5614
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5614, stratify=y, random_state=42) #0.5614 (ser A)

    # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.58441, stratify=y, random_state=42) #0.5614 (ser B)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.682341, stratify=y, random_state=42) #0.5614 (ser C)
    
    X_train_class, X_test_class, Y_train_class, Y_test_class = train_test_split(
        X_class, Y_class, test_size=test_size, random_state=42
    )

    X_train.append(X_train_class)
    X_test.append(X_test_class)
    Y_train.append(Y_train_class)
    Y_test.append(Y_test_class)

X_train = np.vstack(X_train)
X_test = np.vstack(X_test) if X_test else np.array([])  # Nếu không có mẫu test, tạo mảng rỗng
Y_train = np.hstack(Y_train)
Y_test = np.hstack(Y_test) if Y_test else np.array([])

# 4️⃣ Giảm chiều với PCA
print("Đang giảm chiều với PCA...")
pca = PCA(n_components=0.9)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"Số chiều sau PCA: {X_train_pca.shape[1]}")

# 5️⃣ Train SVM
print("Đang huấn luyện mô hình SVM...")
svm = SVC(kernel='linear', C=10, gamma='scale', random_state=42)
svm.fit(X_train_pca, Y_train)  # Đảm bảo dùng dữ liệu đã giảm chiều với PCA
print("Đã huấn luyện xong mô hình SVM!")

# Dự đoán trên tập test
y_pred = svm.predict(X_test_pca)

# Tính độ chính xác
accuracy = accuracy_score(Y_test, y_pred)
print(f"Độ chính xác trên tập test: {accuracy * 100:.2f}%")

#knn
print("Đang huấn luyện mô hình knn...")
knn = KNeighborsClassifier(n_neighbors=10, weights='distance')
knn.fit(X_train_pca, Y_train)  # Đảm bảo dùng dữ liệu đã giảm chiều với PCA
print("Đã huấn luyện xong mô hình knn!")

# Dự đoán trên tập test
y_pred = knn.predict(X_test_pca)

# Tính độ chính xác
accuracy = accuracy_score(Y_test, y_pred)
print(f"Độ chính xác trên tập test: {accuracy * 100:.2f}%")


