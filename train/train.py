import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load dữ liệu đã tiền xử lý
X_train, X_test, Y_train, Y_test, pca = joblib.load("train/preprocessed_data.pkl")

# Đảm bảo thư mục model tồn tại
os.makedirs("model", exist_ok=True)

# 1️⃣ Huấn luyện SVM
svm_model = SVC(kernel="linear", C=1, probability=True, random_state=42)
svm_model.fit(X_train, Y_train)
joblib.dump(svm_model, "model/svm_hog.pkl")

# 2️⃣ Huấn luyện KNN
knn_model = KNeighborsClassifier(n_neighbors=3, weights='distance', p=2)
knn_model.fit(X_train, Y_train)
joblib.dump(knn_model, "model/knn_hog.pkl")

print("trained")

