import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Define the dataset paths
dataset_paths = [
    r"C:\Users\Charan s\Desktop\Dataset\Glass",
    r"C:\Users\Charan s\Desktop\Dataset\Metal",
    r"C:\Users\Charan s\Desktop\Dataset\Paper",
    r"C:\Users\Charan s\Desktop\Dataset\Plastic"
]

# Define class name mapping
CLASS_NAMES = {
    '0': 'Glass',
    '1': 'Metal',
    '2': 'Paper',
    '3': 'Plastic'
}

def load_and_preprocess_data():
    images = []
    labels = []
    
    # Load images and labels from each class directory
    for class_path in dataset_paths:
        class_name = os.path.basename(class_path)
        print(f"Loading images from {class_name}...")
        
        if not os.path.exists(class_path):
            print(f"Warning: Directory {class_path} does not exist!")
            continue
            
        for image_file in os.listdir(class_path):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(class_path, image_file)
                try:
                    # Read and preprocess image
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Failed to load image: {image_path}")
                        continue
                    img = cv2.resize(img, (224, 224))  # Resize to standard size
                    img = img / 255.0  # Normalize pixel values
                    images.append(img)
                    labels.append(class_name)
                except Exception as e:
                    print(f"Error loading image {image_path}: {str(e)}")
    
    if not images:
        raise ValueError("No images were loaded from the dataset!")
    
    print(f"Successfully loaded {len(images)} images from {len(set(labels))} classes")
    return np.array(images), np.array(labels)

def create_cnn_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data()
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Save the label mapping
    label_mapping = {str(i): CLASS_NAMES[str(i)] for i in range(len(label_encoder.classes_))}
    np.save('label_mapping.npy', label_mapping)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Create data augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Create and compile model
    print("Creating and compiling model...")
    model = create_cnn_model(len(np.unique(y_encoded)))
    
    # Use a fixed learning rate with Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Add callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001
        )
    ]
    
    # Train model with data augmentation
    print("Training model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=30,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        steps_per_epoch=len(X_train) // 32
    )
    
    # Save model and label encoder
    model.save('object_detection_model.h5')
    np.save('label_encoder_classes.npy', label_encoder.classes_)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig('training_history.png')
    plt.close()
    
    return model, label_encoder

if __name__ == "__main__":
    train_model() 