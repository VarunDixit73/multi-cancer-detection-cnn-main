import numpy as np
from PIL import Image
import keras

def saved_model(path='skin_cancer/resnet50.json'):
    saved_model  = keras.models.model_from_json(open(path).read())
    return saved_model

def predict(image_path = 'uploads/2.jpg'):
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    img = [read(image_path) ]
    input_data = np.array(img, dtype='uint8')
    input_data = input_data/255

    y_pred = saved_model().predict(input_data)
    result = np.argmax(y_pred, axis=1)[0]
    if result == 1:
        return 'you might have skin cancer'
    else:
        return 'no skin cancer detected'