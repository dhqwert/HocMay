import os
import cv2
import random
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

def preprocess_image(image_path, save_path):
    """ Đọc ảnh, chuyển đổi RGB (nếu cần), resize và lưu. """
    image = cv2.imread(str(image_path))  # Đọc ảnh gốc (có thể RGB hoặc Gray)
    
    # Kiểm tra nếu ảnh là grayscale thì chuyển về RGB
    if len(image.shape) == 2 or image.shape[2] == 1:  
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image_pil = Image.fromarray(image)  # Chuyển sang PIL
    transform = transforms.Compose([
        transforms.Resize((160, 160), antialias=True)
    ])
    
    image_pil = transform(image_pil)  # Resize
    image_pil.save(save_path)  # Lưu ảnh (không dùng torchvision để tránh lỗi màu)

def split_dataset(dataset_dir, output_dir, seed=42):
    random.seed(seed)
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    for split in ["train", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    for user_id in os.listdir(dataset_dir):
        user_path = dataset_dir / user_id
        if not user_path.is_dir():
            continue

        images = list(user_path.glob("*.jpg"))
        random.shuffle(images)
        num_images = len(images)

        if num_images > 20:
            train_count = int(0.8 * num_images)
            test_count = num_images - train_count
        else:
            train_count = int(0.7 * num_images)
            test_count = num_images - train_count

        split_mapping = {
            "train": images[:train_count],
            "test": images[train_count:train_count + test_count]  
        }

        for split, split_images in split_mapping.items():
            split_dir = output_dir / split / user_id
            split_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                save_path = split_dir / img_path.name
                preprocess_image(img_path, save_path)

    
if __name__ == "__main__":
    split_dataset("dataset_aug", "split_dataset")
