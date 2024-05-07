import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
data_dir = 'Data'  # Path to your main data folder
train_dir = 'Data/Train'  # Path to save training data
test_dir = 'Data/Test'  # Path to save test data
test_size = 0.2  # Percentage of data to allocate for testing

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Initialize lists for images and their corresponding labels
images = []
labels = []

# Iterate through each classification folder
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        # List all images in the class directory
        class_images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith('.jpg')]
        images.extend(class_images)
        labels.extend([class_name] * len(class_images))  # Assign label based on folder name

# Check if there are images to split
if len(images) == 0:
    print("No images found in the data directory.")
else:
    # Split images and labels into train and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # Move train images to train directory
    for img, label in zip(train_images, train_labels):
        train_class_dir = os.path.join(train_dir, label)
        os.makedirs(train_class_dir, exist_ok=True)
        shutil.copy(img, os.path.join(train_class_dir, os.path.basename(img)))

    # Move test images to test directory
    for img, label in zip(test_images, test_labels):
        test_class_dir = os.path.join(test_dir, label)
        os.makedirs(test_class_dir, exist_ok=True)
        shutil.copy(img, os.path.join(test_class_dir, os.path.basename(img)))

    print("Data split completed successfully.")
