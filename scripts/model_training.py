import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import argparse

def run(args):
    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir
    os.makedirs(model_dir, exist_ok=True)

    print("----------- Starting Model Training -----------")
    print("Loading preprocessed data...")
    
    # Load preprocessed data
    try:
        X_train = np.load(os.path.join(output_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(output_dir, 'y_train.npy'))
        X_test = np.load(os.path.join(output_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(output_dir, 'y_test.npy'))
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return

    print("Encoding labels...")
    num_classes = 43
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Create validation set
    print("Splitting training data for validation...")
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")

    # Data augmentation
    print("Applying data augmentation...")
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    # Build model
    print("Building model...")
    model = Sequential([
        Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]),
        Conv2D(filters=32, kernel_size=(5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(rate=0.25),

        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(rate=0.25),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(rate=0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=args.learning_rate),
        metrics=['accuracy']
    )
    
    # Train the model
    print("Training the model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=args.batch_size),
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # Save the model and history
    print("Saving model and history...")
    model.save(os.path.join(model_dir, 'best_model.h5'))
    np.save(os.path.join(model_dir, 'history.npy'), history.history)

    print("----------- Model Training Complete -----------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training for Traffic Sign Recognition")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory of the data')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory for outputs')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    args = parser.parse_args()
    run(args)