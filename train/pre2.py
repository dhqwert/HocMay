import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import joblib

# Tùy chọn tập dữ liệu
USE_AUGMENTED_DATA = input("Dùng tập dataset có ảnh tăng cường? (y/n): ").strip().lower() == 'y'

def load_images_from_directory(base_path):
    """Tải dữ liệu từ thư mục chứa ảnh."""
    X, Y = [], []
    for user_id in os.listdir(base_path):
        user_path = os.path.join(base_path, user_id)
        if not os.path.isdir(user_path):
            continue
        for img_name in os.listdir(user_path):
            img = cv2.imread(os.path.join(user_path, img_name), cv2.IMREAD_GRAYSCALE)
            X.append(img)
            Y.append(user_id)
    return np.array(X), np.array(Y)

# 1️⃣ Load dữ liệu
dataset_path = "dataset_aug/" if USE_AUGMENTED_DATA else "dataset_org/"
print(f"Đang tải dữ liệu từ {dataset_path}...")
X, Y = load_images_from_directory(dataset_path)

print(f"Tổng số ảnh: {len(X)}, Số lớp: {len(set(Y))}")

# 2️⃣ Tiền xử lý: trích xuất đặc trưng HOG 
def extract_hog_features(images):
    """Trích xuất đặc trưng HOG."""
    features = [hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2)) for img in images]
    return np.array(features)

X_hog = extract_hog_features(X)
print(f"Feature HOG mỗi ảnh: {X_hog.shape[1]}")

# 3️⃣ Chia tập Train/Test theo từng lớp
X_train, X_test, Y_train, Y_test = [], [], [], []

for label in np.unique(Y):
    X_class = X_hog[Y == label]
    Y_class = Y[Y == label]

    if len(Y_class) == 1:
        X_train.append(X_class)
        Y_train.append(Y_class)
        continue
    
    test_size = 0.2 if len(Y_class) > 20 else 0.3
    
    X_train_class, X_test_class, Y_train_class, Y_test_class = train_test_split(
        X_class, Y_class, test_size=test_size, random_state=42
    )

    X_train.append(X_train_class)
    X_test.append(X_test_class)
    Y_train.append(Y_train_class)
    Y_test.append(Y_test_class)

X_train = np.vstack(X_train)
X_test = np.vstack(X_test) if X_test else np.array([])
Y_train = np.hstack(Y_train)
Y_test = np.hstack(Y_test) if Y_test else np.array([])

# 4️⃣ Giảm chiều với PCA
pca = PCA(n_components=0.7)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"Số chiều sau PCA: {X_train_pca.shape[1]}")

# Lưu dữ liệu đã xử lý
os.makedirs("train", exist_ok=True)
joblib.dump((X_train_pca, X_test_pca, Y_train, Y_test, pca), "train/preprocessed_data.pkl")
print("Đã lưu dữ liệu tiền xử lý vào train/preprocessed_data.pkl")
