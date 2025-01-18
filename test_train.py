import os
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib

matplotlib.use('Agg')


# Load your trained model
model = tf.keras.models.load_model('emotion_model_90_epochs.h5')

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def process_and_predict(image_path):
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print(f"No face detected in {image_path}. Skipping.")
        return None

    for (x, y, w, h) in faces:
        # Extract face
        face = gray[y:y + h, x:x + w]
        resized_face = cv2.resize(face, (48, 48))

        # Normalize and expand dimensions for model input
        normalized_face = np.expand_dims(resized_face, axis=-1) / 255.0
        normalized_face = np.expand_dims(normalized_face, axis=0)

        # Predict emotion
        predictions = model.predict(normalized_face)
        predicted_label = np.argmax(predictions)
        return predicted_label

    return None

def evaluate_model(test_dir):
    y_true = []
    y_pred = []

    for emotion_folder in os.listdir(test_dir):
        emotion_folder_path = os.path.join(test_dir, emotion_folder)
        if not os.path.isdir(emotion_folder_path):
            continue  # Skip files if any

        for filename in os.listdir(emotion_folder_path):
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                file_path = os.path.join(emotion_folder_path, filename)

                # Get the true label from the folder name
                true_label = emotion_folder
                if true_label not in emotion_labels:
                    print(f"Skipping file {filename}: Label '{true_label}' not in emotion labels.")
                    continue

                true_label_idx = emotion_labels.index(true_label)

                predicted_label = process_and_predict(file_path)
                if predicted_label is not None:
                    y_true.append(true_label_idx)
                    y_pred.append(predicted_label)
                else:
                    print(f"Could not predict for file: {filename}")

    if not y_true or not y_pred:
        print("No predictions were made. Please check your test dataset or face detection.")
        return

    # Calculate accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Model Accuracy: {acc * 100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotion_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as 'confusion_matrix.png'")

# Example usage
test_dir = 'C:/Users/febe/PycharmProjects/AI_Emotion_Recognition/test_images'  # Replace with the path to your test_images directory
evaluate_model(test_dir)
