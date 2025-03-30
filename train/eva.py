import os
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Load dữ liệu đã tiền xử lý
X_train, X_test, Y_train, Y_test, pca = joblib.load("train/preprocessed_data.pkl")

# Hàm đánh giá mô hình trên tập test
def evaluate_model(model_path, test_features, test_labels):
    model = joblib.load(model_path)
    predictions = model.predict(test_features)

    print(f"Evaluation for {model_path}:")
    print("Classification Report:")
    print(classification_report(test_labels, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, predictions))
    print("--------------------------------\n")

# Hàm đánh giá mô hình bằng Cross-Validation
def cross_test_model(model_path, features, labels, cv_folds=3):
    model = joblib.load(model_path)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    scores = cross_val_score(model, features, labels, cv=cv, scoring='accuracy')
    print(f"Cross-Validation for {model_path} ({cv_folds}-fold):")
    print(f"Accuracy Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f}")
    print(f"Standard Deviation: {scores.std():.4f}")
    print("--------------------------------\n")

if __name__ == "__main__":
    # Đánh giá SVM model
    print("Evaluating SVM model:")
    evaluate_model("model/svm_hog.pkl", X_test, Y_test)
    cross_test_model("model/svm_hog.pkl", X_train, Y_train)

    # Đánh giá KNN model
    print("Evaluating KNN model:")
    evaluate_model("model/knn_hog.pkl", X_test, Y_test)
    cross_test_model("model/knn_hog.pkl", X_train, Y_train)
