import os
import shutil
from sklearn.model_selection import train_test_split

SOURCE_DIR = 'UTKFace'
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'

for gender in ['male', 'female']:
    os.makedirs(os.path.join(TRAIN_DIR, gender), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, gender), exist_ok=True)

images = [img for img in os.listdir(SOURCE_DIR) if img.endswith('.jpg')]

train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

def move_images(image_list, target_dir):
    for img in image_list:
        try:
            gender_code = img.split('_')[1]
            gender = 'male' if gender_code == '0' else 'female'
            shutil.copy(
                os.path.join(SOURCE_DIR, img),
                os.path.join(target_dir, gender, img)
            )
        except Exception as e:
            print(f"Skipping {img}: {e}")

print("Splitting training set...")
move_images(train_images, TRAIN_DIR)
print("Splitting test set...")
move_images(test_images, TEST_DIR)

print("✅ Done! Dataset organized.")
