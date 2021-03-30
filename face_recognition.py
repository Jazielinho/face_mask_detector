'''
@author: jazielinho

FACE RECOGNITION USING OPEN-CV
'''

import cv2
import numpy as np
from typing import List

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detecta_rostros(frame: np.array, shape) -> np.array:
    ''' Retorna '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) > 0:
        face_crop = []
        for f in faces:
            x, y, w, h = [v for v in f]
            face_crop.append(cv2.resize(frame[y: y + h, x: x + w], shape[:2]))
        return np.concatenate(face_crop)
    return None


if __name__ == '__main__':
    from PIL import Image
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    # video_capture = cv2.VideoCapture(0)
    # ret, frame = video_capture.read()

    # img_path = 'C:/Users/jahaz/OneDrive/Escritorio/download.png'
    # img_path = 'C:/Users/jahaz/OneDrive/Escritorio/foto.JPG'
    img_path = 'C:/Users/jahaz/OneDrive/Escritorio/foto.png'

    image = img_to_array(load_img(img_path)).astype(np.uint8)

    face_crop = detecta_rostros(image, shape=(128, 128, 3))

    for n, face_rect in enumerate(face_crop):
        Image.fromarray(face_rect.astype(np.uint8), 'RGB').show()