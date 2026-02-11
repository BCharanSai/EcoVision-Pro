import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import time
import os

def load_model():
    try:
        # Load the trained model and label encoder
        model = models.load_model('object_detection_model.h5')
        label_encoder_classes = np.load('label_encoder_classes.npy', allow_pickle=True)
        label_mapping = np.load('label_mapping.npy', allow_pickle=True).item()
        return model, label_encoder_classes, label_mapping
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None, None

def setup_camera(camera_id=0):
    """Setup camera with optimized settings for Raspberry Pi"""
    cap = cv2.VideoCapture(camera_id)
    
    # Set lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Set FPS
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    return cap

def real_time_detection():
    # Load model and labels
    model, label_encoder_classes, label_mapping = load_model()
    if model is None:
        print("Failed to load model. Please ensure model files are present.")
        return
    
    # Setup camera
    cap = setup_camera()
    if not cap.isOpened():
        print("Failed to open camera")
        return
    
    # Initialize FPS calculation variables
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    print("Starting real-time detection...")
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Calculate FPS
            frame_count += 1
            if frame_count >= 30:
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()
            
            # Create a copy of the frame for display
            display_frame = frame.copy()
            
            # Preprocess frame for prediction
            processed_frame = cv2.resize(frame, (224, 224))
            processed_frame = processed_frame / 255.0
            processed_frame = np.expand_dims(processed_frame, axis=0)
            
            # Make prediction
            prediction = model.predict(processed_frame, verbose=0)
            predicted_class_idx = np.argmax(prediction)
            predicted_class = label_mapping[str(predicted_class_idx)]
            confidence = np.max(prediction) * 100
            
            # Add a semi-transparent overlay for text background
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            
            # Display prediction with better formatting
            text = f"Material: {predicted_class}"
            confidence_text = f"Confidence: {confidence:.2f}%"
            fps_text = f"FPS: {fps:.1f}"
            
            # Add text with better visibility
            cv2.putText(display_frame, text, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_frame, confidence_text, (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_frame, fps_text, (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add instructions
            cv2.putText(display_frame, "Press 'q' to quit", (10, display_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Material Detection', display_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping detection...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_detection() 