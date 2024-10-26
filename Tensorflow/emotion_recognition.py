import numpy as np
import cv2
from tensorflow.keras.models import load_model
from collections import deque

class EmotionDetector:
    def __init__(self, model_path, emotion_dict):
        self.model = load_model(model_path)
        self.emotion_dict = emotion_dict
        self.window_size = 10
        self.emotion_window = deque(maxlen=self.window_size)
        self.max_emotion = None

    def detect_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = self.model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            emotion = self.emotion_dict[maxindex]

            self.emotion_window.append(prediction[0])

            avg_emotion_probabilities = np.mean(self.emotion_window, axis=0)
            avg_emotion_dict = {self.emotion_dict[i]: avg_emotion_probabilities[i] for i in range(len(self.emotion_dict))}

            cv2.putText(frame, emotion, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return frame, avg_emotion_dict

        return frame, None