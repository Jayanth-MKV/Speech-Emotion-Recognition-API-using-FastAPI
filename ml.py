# ml.py

from PIL import Image
import numpy as np
import io
import tensorflow as tf
from fastapi import HTTPException

model = tf.keras.models.load_model('./model/fer.h5')


async def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('L')
    img = img.resize((48, 48))
    img_array = np.asarray(img) / 255.0
    img_array = img_array.flatten()
    return img_array


async def predict_image(file):
    img_bytes = await file.read()
    img_path = io.BytesIO(img_bytes)
    processed_image = await preprocess_image(img_path)
    processed_image = np.reshape(processed_image, (1, 48, 48, 1))
    predictions = model.predict(processed_image)
    return predictions.tolist()


async def predict_images(files):
    predictions_list = []

    for file in files:
        if not file.content_type.startswith('image'):
            raise HTTPException(
                status_code=400, detail="All files must be images")

        predictions = await predict_image(file)
        ind = np.argmax(predictions[0])
        predictions_list.append(mapper[ind])

    return predictions_list

mapper = {
    0: "happy",
    1: "sad",
    2: "neutral",
}
