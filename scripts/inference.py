import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import glob
import argparse

def run(args):
    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir
    image_path = args.image_path
    process_all = args.process_all
    batch_size = args.batch_size
    os.makedirs(output_dir, exist_ok=True)

    print("----------- Starting Inference -----------")
    print("Loading trained model...")
    
    # Load the trained model
    model_path = os.path.join(model_dir, 'best_model.h5')
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Prepare to save predictions
    result_path = os.path.join(output_dir, 'inference_results.txt')
    predictions = []

    if process_all:
        # Process all PNG files in data/Test/
        test_dir = os.path.join(data_dir, 'Test')
        image_files = glob.glob(os.path.join(test_dir, '*.png'))
        if not image_files:
            print(f"Error: No PNG files found in {test_dir}")
            return
        print(f"Found {len(image_files)} PNG files in {test_dir}")

        # Load and preprocess images in batches
        batch_images = []
        batch_filenames = []
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error loading image {img_path}, skipping")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_NEAREST)
            img = img.astype('float32') / 255.0
            batch_images.append(img)
            batch_filenames.append(os.path.basename(img_path))

            # Process the batch when it reaches batch_size
            if len(batch_images) == batch_size:
                print(f"Performing inference on batch of {len(batch_images)} images...")
                batch_array = np.array(batch_images)  # Shape: (batch_size, 30, 30, 3)
                batch_predictions = model.predict(batch_array, verbose=1)
                predicted_classes = np.argmax(batch_predictions, axis=1)
                for filename, pred_class in zip(batch_filenames, predicted_classes):
                    print(f"Image: {filename}, Predicted class: {pred_class}")
                    predictions.append(f"Image: {filename}, Predicted class: {pred_class}")
                batch_images = []
                batch_filenames = []

        # Process remaining images (if any)
        if batch_images:
            print(f"Performing inference on remaining {len(batch_images)} images...")
            batch_array = np.array(batch_images)  # Shape: (remaining, 30, 30, 3)
            batch_predictions = model.predict(batch_array, verbose=1)
            predicted_classes = np.argmax(batch_predictions, axis=1)
            for filename, pred_class in zip(batch_filenames, predicted_classes):
                print(f"Image: {filename}, Predicted class: {pred_class}")
                predictions.append(f"Image: {filename}, Predicted class: {pred_class}")
    else:
        # Process a single image
        print("Loading and preprocessing image...")
        if not os.path.exists(image_path):
            print(f"Error: Image {image_path} not found")
            return
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image {image_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_NEAREST)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)  # Shape: (1, 30, 30, 3)

        # Perform inference
        print("Performing inference...")
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        print(f"Predicted class: {predicted_class}")
        predictions.append(f"Image: {os.path.basename(image_path)}, Predicted class: {predicted_class}")

    # Save all predictions
    with open(result_path, 'w') as f:
        for pred in predictions:
            f.write(pred + '\n')
    print(f"Inference results saved to {result_path}")

    print("----------- Inference Complete -----------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Traffic Sign Recognition")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory of the data')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory for outputs')
    parser.add_argument('--image_path', type=str, default='data/Test/00000.png', help='Path to the image for inference (used if --process_all is not set)')
    parser.add_argument('--process_all', action='store_true', help='Process all PNG files in data/Test/ directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference when processing all images')
    args = parser.parse_args()
    run(args)