import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Define the dataset paths
BASE_DATASET_DIR = r"C:\Users\Charan s\Desktop\Dataset"
dataset_paths = [
    os.path.join(BASE_DATASET_DIR, d)
    for d in os.listdir(BASE_DATASET_DIR)
    if os.path.isdir(os.path.join(BASE_DATASET_DIR, d))
]

def load_and_preprocess_data():
    images = []
    labels = []

    # Here, each path in dataset_paths is expected to be a class folder
    # under BASE_DATASET_DIR (e.g., Glass, Metal, Paper, Plastic)
    for class_path in dataset_paths:
        class_name = os.path.basename(class_path)

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
        raise ValueError(
            f"No images were loaded from dataset directory: {BASE_DATASET_DIR}. "
            f"Make sure it contains class folders (e.g., Glass, Metal, Paper, Plastic) with image files."
        )

    return np.array(images), np.array(labels)

def create_cnn_model(num_classes):
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Fourth Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
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
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Create and compile model
    print("Creating and compiling model...")
    model = create_cnn_model(len(np.unique(y_encoded)))
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Train model
    print("Training model...")
    history = model.fit(X_train, y_train,
                       epochs=20,
                       batch_size=32,
                       validation_data=(X_test, y_test))
    
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

def real_time_detection():
    # Load the trained model and label encoder
    model = models.load_model('object_detection_model.h5')
    label_encoder_classes = np.load('label_encoder_classes.npy', allow_pickle=True)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame
        processed_frame = cv2.resize(frame, (224, 224))
        processed_frame = processed_frame / 255.0
        processed_frame = np.expand_dims(processed_frame, axis=0)
        
        # Make prediction
        prediction = model.predict(processed_frame)
        predicted_class = label_encoder_classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        # Display prediction
        cv2.putText(frame, f"{predicted_class}: {confidence:.2f}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Object Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Train the model
    model, label_encoder = train_model()
    
    # Start real-time detection
    print("Starting real-time detection...")
    real_time_detection() 