

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
            img_ = np.reshape(img_, (1, *config_tr.SHAPE))
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
