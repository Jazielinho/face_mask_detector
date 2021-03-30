'''
@author: jazielinho

FACE RECOGNITION USING OPEN-CV
'''

import cv2
import numpy as np
from typing import List, Tuple

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detecta_rostros(frame: np.array) -> Tuple[List, List]:
    ''' Retorna '''
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.3,
        minNeighbors=5,
        # minSize=(30, 30),
        # flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) > 0:
        face_crop = []
        list_ubications = []
        for x, y, w, h in faces:
            face_crop.append(frame[y: y + h, x: x + w])
            list_ubications.append([x, y, w, h])
        return face_crop, list_ubications
    return [], []


if __name__ == '__main__':
    from PIL import Image
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()

    # img_path = 'C:/Users/jahaz/OneDrive/Escritorio/download.png'
    # img_path = 'C:/Users/jahaz/OneDrive/Escritorio/foto.JPG'
    # img_path = 'C:/Users/jahaz/OneDrive/Escritorio/BEAUTIFUL-FACES_3249636b.jpg'
    # img_path = 'C:/Users/jahaz/OneDrive/Escritorio/descubre-como-tener-un-rostro-10-foto-freepik.jpeg'

    # image = img_to_array(load_img(img_path)).astype(np.uint8)
    #
    image = frame

    face_crop = detecta_rostros(image, shape=(224, 224, 3))

    for n, face_rect in enumerate(face_crop):
        Image.fromarray(face_rect.astype(np.uint8), 'RGB').show()