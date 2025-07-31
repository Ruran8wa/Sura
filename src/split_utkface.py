import os
import shutil
from sklearn.model_selection import train_test_split

# Set paths
SOURCE_DIR = 'UTKFace'  # Replace with your dataset folder name
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'

# Make sure destination folders exist
for gender in ['male', 'female']:
    os.makedirs(os.path.join(TRAIN_DIR, gender), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, gender), exist_ok=True)

# Get all image filenames
images = [img for img in os.listdir(SOURCE_DIR) if img.endswith('.jpg')]

# Shuffle and split dataset
train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

# Helper to move files based on gender
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

# Move files
print("Splitting training set...")
move_images(train_images, TRAIN_DIR)
print("Splitting test set...")
move_images(test_images, TEST_DIR)

print("✅ Done! Dataset organized.")
