import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import argparse

def run(args):
    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("----------- Starting Model Evaluation -----------")
    print("Loading preprocessed test data...")
    
    # Load preprocessed test data
    try:
        X_test = np.load(os.path.join(output_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(output_dir, 'y_test.npy'))
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return

    print("Encoding test labels...")
    num_classes = 43
    y_test = to_categorical(y_test, num_classes)

    # Load the trained model
    print("Loading trained model...")
    model_path = os.path.join(model_dir, 'best_model.h5')
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Evaluate the model
    print("Evaluating model on test data...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save evaluation results
    results_path = os.path.join(output_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Test Loss: {loss:.4f}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
    print(f"Evaluation results saved to {results_path}")

    print("----------- Model Evaluation Complete -----------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation for Traffic Sign Recognition")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory of the data')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory for outputs')
    args = parser.parse_args()
    run(args)