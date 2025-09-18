import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib

# --- 1. Define Constants and Paths ---
DATA_DIR = pathlib.Path('data/casting_data/train')
IMAGE_HEIGHT = 150
IMAGE_WIDTH = 150
BATCH_SIZE = 32
MODEL_SAVE_PATH = 'defect_detector.h5'

def train_and_save_model():
    """
    Loads image data, builds, trains, and saves a CNN model for defect detection.
    """
    print("Starting model training process...")

    # --- 2. Load and Prepare the Dataset ---
    # Keras utility to load images from directories. It automatically infers labels
    # from the folder names ('def_front', 'ok_front').
    # We use a 80/20 split for training and validation.
    print(f"Loading images from: {DATA_DIR}")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"Found classes: {class_names}")

    # --- 3. Build the CNN Model ---
    # A simple sequential model is sufficient for this binary classification task.
    print("Building the CNN model architecture...")
    model = keras.Sequential([
        # Rescale pixel values from [0, 255] to [0, 1]
        layers.Rescaling(1./255, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        
        # First convolutional block
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        # Second convolutional block
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        # Third convolutional block
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        # Flatten the results to feed into a dense layer
        layers.Flatten(),
        
        # A standard dense layer
        layers.Dense(128, activation='relu'),
        
        # Output layer. Since we have 2 classes, we only need 1 output neuron.
        layers.Dense(len(class_names) -1, activation='sigmoid')
    ])

    # --- 4. Compile the Model ---
    # Configure the model for training.
    print("Compiling the model...")
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    
    model.summary()

    # --- 5. Train the Model ---
    print("Starting training... This might take a few minutes.")
    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # --- 6. Save the Trained Model ---
    print(f"Training complete. Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("Model saved successfully!")
    print("--------------------------------------------------")


if __name__ == '__main__':
    train_and_save_model()