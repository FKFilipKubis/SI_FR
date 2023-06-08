# import os
# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from keras.utils import to_categorical
#
# # Definicja ścieżki do foldera zawierającego zdjęcia
# path = 'H:/archive/fruits-360_dataset/fruits-360/Test/full_set'
#
# # Inicjalizacja pustych list na obrazy i etykiety
# images = []
# labels = []
#
# # Przetwarzanie zdjęć i zbieranie danych treningowych
# classes = ['Apple', 'Banana', 'Cherry']
# for i, fruit_class in enumerate(classes):
#     class_path = os.path.join(path, fruit_class)
#     for image_name in os.listdir(class_path):
#         image_path = os.path.join(class_path, image_name)
#         image = cv2.imread(image_path)
#         image = cv2.resize(image, (512, 512))
#         images.append(image)
#         labels.append(i)
#
#
# # Konwersja list na tablice
# images = np.array(images)
# labels = np.array(labels)
#
# # Podział danych na zbiory treningowe i testowe
# train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
# num_classes = len(classes)
# train_labels = to_categorical(train_labels, num_classes)
# test_labels = to_categorical(test_labels, num_classes)
#
# # Tworzenie modelu CNN
# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # Trenowanie modelu
# model.fit(train_images, train_labels, epochs=10, batch_size=16)
#
# # Ocena modelu na danych testowych
# test_loss, test_accuracy = model.evaluate(test_images, test_labels)
# predicted_labels = model.predict(test_images)
#
#
# # Zapis modelu do pliku
# current_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(current_dir, 'model.h5')
# model.save(model_path)

import cv2
import numpy as np
import os
from keras.models import load_model

# wczytanie wytrenowanego modelu
model_path = 'C:/Users/FK_PC/PycharmProjects/SI_CNN/model.h5'
model = load_model(model_path)

# Definicja klas owoców
classes = ['Apple', 'Banana', 'Cherry']

# rozpoczęcie przechwytywania z kamery
cap = cv2.VideoCapture(0)

# funkcja predykująca aktualny owoc lub warzywo
def predict_fruit(image):
    # zmiana rozmiaru na akceptowalny przez model
    image = cv2.resize(image, (512, 512))
    image = image.reshape(1, 512, 512, 3)

    # wykonywanie predykcji ze zdjęcia
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = classes[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    confidence_percentage = round(confidence * 100, 2)
    return predicted_class, confidence_percentage

# ścieżka do folderu z zapisywanymi zdjęciami
image_folder = 'C:/Users/SPANKO/PycharmProjects/SI_CNN/camera_capture'
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

i = 0
# odczyt z kamery
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)

    # zapisywanie zdjęc do folderu
    cv2.imwrite(os.path.join(image_folder, f'frame{i}.jpg'), frame)
    i += 1

    # predykcja jaki to owoc lub warzywo
    predicted_class, confidence = predict_fruit(frame)
    text = f"Class: {predicted_class}, Confidence: {confidence}%"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# zamknięcie okien i wyłącznie kamery
cap.release()
cv2.destroyAllWindows()
