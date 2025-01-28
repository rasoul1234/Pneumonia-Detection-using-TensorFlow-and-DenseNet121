import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

def main():
    # Check if GPU is available
    device_name = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
    print(f"Using device: {device_name}")

    # Dataset paths
    base_dir = r"C:\Users\Muhammad Rasoul\Downloads\Pneumonia_Detector-main\chest_xray-Dataset\chest_xray"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")

    # Image preprocessing and augmentation
    image_size = (224, 224)
    batch_size = 32

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True
    )
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical"
    )
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical"
    )
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    # Build the DenseNet121 model
    base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    output_layer = Dense(2, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output_layer)

    # Freeze the base model
    base_model.trainable = False

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Train the model
    def train_model(model, train_generator, val_generator, epochs=5):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            print("-" * 20)

            # Training phase
            start_time = time.time()
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=1,
                verbose=1
            )

            train_loss = history.history["loss"][0]
            train_acc = history.history["accuracy"][0]
            val_loss = history.history["val_loss"][0]
            val_acc = history.history["val_accuracy"][0]

            epoch_time = time.time() - start_time
            print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            print(f"Epoch time: {epoch_time:.2f}s")

    train_model(model, train_generator, val_generator, epochs=5)

    # Save the model
    model.save("pneumonia_densenet121.h5")
    print("Model saved as 'pneumonia_densenet121.h5'.")

    # Evaluate the model
    def evaluate_model(model, test_generator):
        test_loss, test_acc = model.evaluate(test_generator, verbose=1)
        print(f"Test Accuracy: {test_acc:.4f}")

    evaluate_model(model, test_generator)

if __name__ == "__main__":
    main()
