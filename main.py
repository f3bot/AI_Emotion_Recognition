import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image_dataset_from_directory
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def load_fer2013(data_dir):
    train_dir = f"{data_dir}/train"
    test_dir = f"{data_dir}/test"

    train_ds = image_dataset_from_directory(
        train_dir,
        image_size=(48, 48),
        color_mode='grayscale',
        batch_size=64
    )
    test_ds = image_dataset_from_directory(
        test_dir,
        image_size=(48, 48),
        color_mode='grayscale',
        batch_size=64
    )

    train_ds = train_ds.map(normalize_img)
    test_ds = test_ds.map(normalize_img)

    return train_ds, test_ds

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_ds, test_ds):
    history = model.fit(train_ds, validation_data=test_ds, epochs=45)
    model.save('emotion_model_nowy_dzialajacy__.h5')
    return history

def plot_training_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

def load_model():
    return tf.keras.models.load_model('emotion_model_nowy_dzialajacy.h5')

def predict_emotion():
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    model = load_model()

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            resized_face = cv2.resize(face, (48, 48))
            normalized_face = np.expand_dims(resized_face, axis=-1) / 255.0
            normalized_face = np.expand_dims(normalized_face, axis=0)

            predictions = model.predict(normalized_face)
            emotion = emotion_labels[np.argmax(predictions)]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    data_dir = 'C:/Users/febe/PycharmProjects/AI_Emotion_Recognition'

    # Load the training and testing datasets
    train_ds, test_ds = load_fer2013(data_dir)

    # Build the model
    model = build_model()

    # Train the model for 90 epochs
    history = train_model(model, train_ds, test_ds)

    # Save the trained model to a new file
    model.save('emotion_model_90_epochs.h5')

    # Optionally, plot the training results
    plot_training_results(history)