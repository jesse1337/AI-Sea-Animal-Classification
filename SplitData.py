import os
import shutil
from sklearn.model_selection import train_test_split

DATA = 'Data'
TRAIN = 'Data/Train'
TEST = 'Data/Test'

# 80-20% split between training and testing
test_size = 0.2

# If directories don't exist, create them
os.makedirs(TRAIN, exist_ok=True)
os.makedirs(TEST, exist_ok=True)

img = []
labels = []


# Go through everything and acquire data (img/label)
for specie in os.listdir(DATA):
    specie_folder = os.path.join(DATA, specie)

    if os.path.isdir(specie_folder):
        specie_image = [os.path.join(specie_folder, img) for img in os.listdir(
            specie_folder) if img.endswith('.jpg')]
        img.extend(specie_image)

        labels.extend([specie] * len(specie_image))

# Check if image exists.
if len(img) == 0:
    print("NO IMAGES IN THIS FOLDER")
else:
    train_img, test_img, train_labels, test_labels = train_test_split(
        img, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # Move train img to train directory
    for img, label in zip(train_img, train_labels):
        train_specie_folder = os.path.join(TRAIN, label)
        os.makedirs(train_specie_folder, exist_ok=True)
        shutil.copy(img, os.path.join(
            train_specie_folder, os.path.basename(img)))

    # Move test img to test directory
    for img, label in zip(test_img, test_labels):
        test_specie_folder = os.path.join(TEST, label)
        os.makedirs(test_specie_folder, exist_ok=True)
        shutil.copy(img, os.path.join(
            test_specie_folder, os.path.basename(img)))

    print("DATA SPLIT DONE")
