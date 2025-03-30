import os
import numpy as np
import torch
import joblib
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load FaceNet model (pretrained)
model = InceptionResnetV1(pretrained="vggface2").eval()

# Transform (chỉ normalize, không resize vì ảnh đã chuẩn)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Hàm load và xử lý ảnh (KHÔNG resize, KHÔNG đổi màu)
def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0)  # Thêm batch dimension
    except:
        return None  # Bỏ qua ảnh lỗi

# Hàm trích xuất đặc trưng từ FaceNet
def extract_features(image_folder):
    features, labels = [], []
    if not os.path.exists(image_folder):
        return np.array([]), np.array([])

    for user_id in os.listdir(image_folder):
        user_path = os.path.join(image_folder, user_id)
        if not os.path.isdir(user_path):
            continue
        for img_name in os.listdir(user_path):
            img_tensor = preprocess_image(os.path.join(user_path, img_name))
            if img_tensor is None:
                continue
            with torch.no_grad():
                embedding = model(img_tensor).cpu().numpy().flatten()
            features.append(embedding)
            labels.append(user_id)

    return np.array(features), np.array(labels)

# Load dataset
train_features, train_labels = extract_features("split_dataset/train")
# test_features, test_labels = extract_features("split_dataset/test")

# Nếu tập train rỗng thì dừng
if train_features.size == 0:
    exit(1)

# Đảm bảo thư mục model tồn tại
os.makedirs("model", exist_ok=True)

# Huấn luyện và lưu mô hình
svm_model = SVC(kernel="linear", C=1, probability=True, random_state=42)
svm_model.fit(train_features, train_labels)
joblib.dump(svm_model, "model/svm_facenet.pkl")

knn_model = KNeighborsClassifier(n_neighbors=3, weights='distance', p=2)
knn_model.fit(train_features, train_labels)
joblib.dump(knn_model, "model/knn_facenet.pkl")


print("trained")
