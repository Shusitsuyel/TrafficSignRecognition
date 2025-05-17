import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import argparse

def run(args):
    data_dir = args.data_dir
    train_dir = os.path.join(data_dir, 'Train')
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("----------- Starting Data Preprocessing -----------")
    print("Loading training images...")
    
    # Load dataset
    data = []
    labels = []
    classes = 43
    
    for label in range(classes):
        label_dir = os.path.join(train_dir, str(label))
        if not os.path.exists(label_dir):
            print(f"Warning: Directory {label_dir} not found, skipping")
            continue
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            img = cv2.imread(img_path, -1)
            if img is None:
                print(f"Warning: Failed to load image {img_path}, skipping")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_NEAREST)
            data.append(img)
            labels.append(label)

    if not data:
        print("Error: No valid images loaded")
        return

    data = np.array(data)
    labels = np.array(labels)
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    
    # Splitting training and testing dataset
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Balance training classes
    print("Balancing training classes...")
    unique, counts = np.unique(y_train, return_counts=True)
    print("Class distribution before balancing:", dict(zip(unique, counts)))
    
    ros = RandomOverSampler(random_state=42)
    X_reshaped = X_train.reshape(len(X_train), -1)
    X_balanced, y_balanced = ros.fit_resample(X_reshaped, y_train)
    X_train = X_balanced.reshape(-1, 30, 30, 3)
    y_train = y_balanced
    print("Class distribution after balancing:", dict(zip(*np.unique(y_train, return_counts=True))))
    
    # Normalize images
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Save preprocessed data
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    print(f"Preprocessed data saved to {output_dir}")

    print("----------- Data Preprocessing Complete -----------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing for Traffic Sign Recognition")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory of the data')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory for outputs')
    args = parser.parse_args()
    run(args)