

import os
from tensorflow import keras
from PIL import Image
import numpy as np

model_path = r"D:\disease_prediction\sample_dataset\skin_cancer_model.h5"
model = keras.models.load_model(model_path, compile=False)

test_folder = r"D:\disease_prediction\sample_dataset\test_images"

class_labels = ['basal_cell_carcinoma', 'melanoma', 'nevus']

if not os.path.exists(test_folder):
    raise FileNotFoundError(f"Test images folder not found: {test_folder}")

for file_name in os.listdir(test_folder):
    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        file_path = os.path.join(test_folder, file_name)
        print(f"Testing on: {file_name}")

        img = Image.open(file_path).resize((224, 224))
        img_array = np.array(img) / 255.0

        if img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[..., :3]

        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        print(f"Prediction: {predicted_class} with confidence {confidence:.4f}\n")
