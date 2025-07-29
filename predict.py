import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# 🔽 Параметри
img_size = (224, 224)
test_folder = "data/catvsdog100/test/"  # Папка з зображеннями
model_path = "models/fine_tuned_model.keras"
class_names = sorted(os.listdir("data/catvsdog100/train/"))  # ['cats', 'dogs']

# 📦 Завантаження моделі
model = load_model(model_path)

# 📸 Отримання зображень з папки
image_files = [
    f for f in os.listdir(test_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# 🔍 Обробка і передбачення
for filename in image_files:
    img_path = os.path.join(test_folder, filename)

    # Завантаження зображення
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Передбачення
    pred = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(pred)]

    print(f"{filename}: {predicted_class}")
