import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from traindl import extract_features  # Import hàm trích xuất đặc trưng

def evaluate_model(model_path, test_features, test_labels):
    model = joblib.load(model_path)
    predictions = model.predict(test_features)
    
    print(f"Evaluation for {model_path}:")
    print("Classification Report:")
    print(classification_report(test_labels, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, predictions))
    print("--------------------------------\n")

def cross_test_model(model_path, features, labels, cv_folds=3):
    """ Thực hiện đánh giá Cross-Validation trên mô hình """
    model = joblib.load(model_path)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    scores = cross_val_score(model, features, labels, cv=cv, scoring='accuracy')
    print(f"Cross-Validation for {model_path} ({cv_folds}-fold):")
    print(f"Accuracy Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f}")
    print(f"Standard Deviation: {scores.std():.4f}")
    print("--------------------------------\n")

def plot_label_distribution(labels, title):
    plt.figure(figsize=(12, 6))
    sns.countplot(x=labels, color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout() 
    plt.show()
    
if __name__ == "__main__":
    # Load test dataset
    test_features, test_labels = extract_features("split_dataset/test")
    test_features, test_labels = extract_features("split_dataset/test")
    train_features, train_labels = extract_features("split_dataset/train")

    # Đánh giá SVM model
    print("Evaluating SVM model:")
    evaluate_model("model/svm_facenet.pkl", test_features, test_labels)
    cross_test_model("model/svm_facenet.pkl", test_features, test_labels)

    # Đánh giá KNN model
    print("Evaluating KNN model:")
    evaluate_model("model/knn_facenet.pkl", test_features, test_labels)
    cross_test_model("model/knn_facenet.pkl", test_features, test_labels)

    plot_label_distribution(train_labels, "Phân bố số lượng mẫu tập train")
    plot_label_distribution(test_labels, "Phân bố số lượng mẫu tập test")
