import numpy as np
import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, make_scorer, precision_score, recall_score, f1_score
from sklearn.datasets import fetch_lfw_people

# Tải dữ liệu LFW
lfw = fetch_lfw_people(min_faces_per_person=10, color=True, resize=1.0)
X, y = lfw.images, lfw.target
class_names = lfw.target_names

# Chuyển đổi ảnh sang tensor PyTorch
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

X = torch.stack([transform(img) for img in X])

# Mã hóa nhãn
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Chia train/test (không chia val)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5614, stratify=y, random_state=42) #0.5614 (ser A)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.58441, stratify=y, random_state=42) #0.5614 (ser B)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.682341, stratify=y, random_state=42) #0.5614 (ser C)


# Thống kê dữ liệu
num_classes = len(class_names)
num_train_samples = len(X_train)
num_test_samples = len(X_test)

print(f"Number of classes: {num_classes}")
print(f"Total training images: {num_train_samples}")
print(f"Total testing images: {num_test_samples}")

# Load pre-trained FaceNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Function to extract embeddings
def get_embeddings(images):
    images = images.to(device)
    with torch.no_grad():
        embeddings = facenet(images)
    return embeddings.cpu().numpy()

# Extract embeddings
X_train_embed = get_embeddings(X_train)
X_test_embed = get_embeddings(X_test)

# Reduce dimensions using PCA
# pca = PCA(n_components=128)
# X_train_pca = pca.fit_transform(X_train_embed)
# X_test_pca = pca.transform(X_test_embed)

# Train SVM
svm = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm.fit(X_train_embed, y_train)
y_pred_svm = svm.predict(X_test_embed)
svm_acc = accuracy_score(y_test, y_pred_svm)
print(f"Accuracy (FaceNet + SVM): {svm_acc:.4f}")

# Train KNN
knn = KNeighborsClassifier(n_neighbors=10, weights='distance')
knn.fit(X_train_embed, y_train)
y_pred_knn = knn.predict(X_test_embed)
knn_acc = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy (FaceNet + KNN): {knn_acc:.4f}")

# Cross-validation metrics
y_pred_cv = cross_val_predict(svm, X_train_embed, y_train, cv=10)
print("=== Classification Report ===")
print(classification_report(y_train, y_pred_cv, target_names=class_names))

scoring = {
    'precision': make_scorer(precision_score, average='macro'),
    'recall': make_scorer(recall_score, average='macro'),
    'f1': make_scorer(f1_score, average='macro')
}

scores = cross_validate(svm, X_train_embed, y_train, cv=10, scoring=scoring)

print(f"Precision: {scores['test_precision'].mean():.4f}")
print(f"Recall: {scores['test_recall'].mean():.4f}")
print(f"F1-score: {scores['test_f1'].mean():.4f}")

# Cross-validation metrics
y_pred_cv = cross_val_predict(knn, X_train_embed, y_train, cv=10)
print("=== Classification Report ===")
print(classification_report(y_train, y_pred_cv, target_names=class_names))

scoring = {
    'precision': make_scorer(precision_score, average='macro'),
    'recall': make_scorer(recall_score, average='macro'),
    'f1': make_scorer(f1_score, average='macro')
}

scores = cross_validate(knn, X_train_embed, y_train, cv=10, scoring=scoring)

print(f"Precision: {scores['test_precision'].mean():.4f}")
print(f"Recall: {scores['test_recall'].mean():.4f}")
print(f"F1-score: {scores['test_f1'].mean():.4f}")
