'''
@author: jazielinho

FACE RECOGNITION USING OPEN-CV
'''

import cv2
import numpy as np
from typing import List, Tuple

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detecta_rostros(frame: np.array) -> Tuple[List, List]:
    ''' Retorna lista con imágenes de rostros y su ubicación en la imagen original '''
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.1,
        minNeighbors=5,
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

    # video_capture = cv2.VideoCapture(0)
    # ret, frame = video_capture.read()

    img_path = 'C:/Users/jahaz/OneDrive/Escritorio/Rostros-reveladores...-¿Qué-dice-el-tuyo.jpg'

    image = img_to_array(load_img(img_path)).astype(np.uint8)

    face_crop, list_ubications = detecta_rostros(image)

    for n, face_rect in enumerate(face_crop):
        Image.fromarray(face_rect.astype(np.uint8), 'RGB').show()