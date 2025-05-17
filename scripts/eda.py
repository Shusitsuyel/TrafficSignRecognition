import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import argparse

def run(args):
    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("----------- Starting Exploratory Data Analysis (EDA) -----------")
    print("Loading preprocessed labels...")
    
    # Load the preprocessed training labels
    try:
        y_train = np.load(os.path.join(output_dir, 'y_train.npy'))
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return
    
    # Class distribution plot
    classes = 43
    class_counts = np.bincount(y_train, minlength=classes)
    class_labels = np.arange(classes)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=class_labels, y=class_counts)
    plt.title("Class Distribution of Traffic Signs")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()
    print(f"Class distribution plot saved to {output_dir}/class_distribution.png")

    # Load sample images for visualization
    print("Loading sample images...")
    sample_images = []
    sample_labels = []
    
    for label in range(classes):
        label_dir = os.path.join(data_dir, 'Train', str(label))
        if not os.path.exists(label_dir):
            print(f"Warning: Directory {label_dir} not found, skipping")
            continue
        img_files = os.listdir(label_dir)
        if not img_files:
            print(f"Warning: No images found in {label_dir}, skipping")
            continue
        img_path = os.path.join(label_dir, img_files[0])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load image {img_path}, skipping")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sample_images.append(img)
        sample_labels.append(label)
    
    if not sample_images:
        print("Error: No sample images loaded for visualization")
        return
    
    # Plot sample images
    plt.figure(figsize=(20, 12))
    for i in range(len(sample_images)):
        plt.subplot(5, 9, i + 1)
        plt.imshow(sample_images[i])
        plt.title(f"Class: {sample_labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_images.png'))
    plt.close()
    print(f"Sample images plot saved to {output_dir}/sample_images.png")
    
    print("----------- EDA Complete -----------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis for Traffic Sign Recognition")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory of the data')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory for outputs')
    args = parser.parse_args()
    run(args)