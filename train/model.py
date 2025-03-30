import os
import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Tải dữ liệu tiền xử lý
print("Đang tải dữ liệu tiền xử lý...")
X_train_pca, X_test_pca, Y_train, Y_test, pca = joblib.load("train/preprocessed_data.pkl")

# 2️⃣ Train SVM
svm = SVC(kernel='linear', C=10, gamma='scale', random_state=42)
svm.fit(X_train_pca, Y_train)
print("Đã huấn luyện xong mô hình SVM!")

# Cross-validation SVM
cv_scores = cross_val_score(svm, X_train_pca, Y_train, cv=10)
print(f"Cross-Validation Accuracy (SVM): {cv_scores.mean() * 100:.2f}%")

# 3️⃣ Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, Y_train)
print("Đã huấn luyện xong mô hình KNN!")

# Cross-validation KNN
cv_scores = cross_val_score(knn, X_train_pca, Y_train, cv=10)
print(f"Cross-Validation Accuracy (KNN): {cv_scores.mean() * 100:.2f}%")

# 4️⃣ Đánh giá mô hình
for model, name in [(svm, "SVM"), (knn, "KNN")]:
    y_pred = model.predict(X_test_pca)
    cm = confusion_matrix(Y_test, y_pred)
    print(f"{name}:")
    print(f"Accuracy: {accuracy_score(Y_test, y_pred) * 100:.2f}%")
    print(f"F1 Score: {f1_score(Y_test, y_pred, average='weighted') * 100:.2f}%")
    print(f"Recall: {recall_score(Y_test, y_pred, average='weighted') * 100:.2f}%")
    print(f"Precision: {precision_score(Y_test, y_pred, average='weighted') * 100:.2f}%")
    print("\nBáo cáo phân loại:\n", classification_report(Y_test, y_pred))
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - Test Set ({name})")
    plt.show()

# 5️⃣ Lưu mô hình
os.makedirs("model", exist_ok=True)
joblib.dump(svm, "model/model_svm.pkl")
joblib.dump(pca, "model/model_pca.pkl")
joblib.dump(knn, "model/model_knn.pkl")
print("Đã lưu mô hình vào model_svm.pkl, model_pca.pkl và model_knn.pkl")
