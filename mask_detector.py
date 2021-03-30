'''
Autor: Jazielinho
'''


import face_recognition
import numpy as np
import cv2
import tensorflow as tf
from training import config_tr

from typing import List, Tuple

import config

model = None

cap = cv2.VideoCapture(0)

color_dict = {1: (0, 255, 0), 0: (0, 0, 255)}
labels_dict = {0: 'no mascara', 1: 'mascara'}


def load_model() -> tf.keras.Model:
    ''' Cargando el modelo '''
    global model
    if model is None:
        model_file = open(config_tr.MODEL_PATH_JSON, 'r')
        model = model_file.read()
        model_file.close()
        model = tf.keras.models.model_from_json(model)
        model.load_weights(config_tr.MODEL_PATH_H5)
    return model


def prepara_imagen_array(img: np.ndarray) -> Tuple[List, List]:
    ''' De la imagen capturara, extrae los rostros y preprocesa '''
    face_crop, list_ubications = face_recognition.detecta_rostros(frame=img)
    face_to_predict = []

    if len(face_crop) > 0:
        for face_ in face_crop:
            img_ = cv2.resize(face_, config_tr.SHAPE[:2])
            img_ = np.reshape(img_, (1, *config_tr.SHAPE, 3))
            img_ = tf.keras.applications.mobilenet.preprocess_input(img_)
            face_to_predict.append(img_)
    return face_to_predict, list_ubications


def get_predictions(face_to_predict: List) -> List:
    global model
    model = load_model()

    list_clases = []
    for face_ in face_to_predict:
        prob = model.predict(face_).ravel()
        if prob > 0.5:
            list_clases.append(0)
        else:
            list_clases.append(1)

    return list_clases


def main():
    while True:
        sucess, img = cap.read()
        face_to_predict, list_ubications = prepara_imagen_array(img=img)
        list_clases = get_predictions(face_to_predict=face_to_predict)

        if len(list_clases) > 0:

            for enum in range(len(list_clases)):
                x, y, w, h = list_ubications[enum]
                cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[list_clases[enum]], 2)
                cv2.rectangle(img, (x, y - 40), (x+w, y), color_dict[list_clases[enum]], -2)
                cv2.putText(img, labels_dict[list_clases[enum]], (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.75,
                            (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Result", img)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break


if __name__ == '__main__':
    main()

