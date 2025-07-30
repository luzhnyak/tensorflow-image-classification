import os
import numpy as np
from PIL import Image
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection

from keras.models import load_model
from keras.preprocessing import image as keras_image
from keras.applications.mobilenet_v2 import preprocess_input

# Завантаження моделей
seg_model_id = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(seg_model_id)
det_model = DetrForObjectDetection.from_pretrained(seg_model_id)
clf_model = load_model("models/fine_tuned_model.keras")

# Класи
class_names = sorted(os.listdir("data/catvsdog100/train/"))

# Папка з зображеннями
input_folder = "data/catvsdog100/test/"

for img_file in os.listdir(input_folder):
    if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(input_folder, img_file)
    pil_img = Image.open(img_path).convert("RGB")

    # Детекція
    inputs = processor(images=pil_img, return_tensors="pt")
    outputs = det_model(**inputs)

    # Отримуємо bounding boxes з ймовірністю > 0.9
    target_sizes = torch.tensor([pil_img.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.9
    )[0]

    if len(results["scores"]) == 0:
        print(f"{img_file}: Об'єкт не знайдено")
        continue

    # Вирізаємо перший знайдений об'єкт
    box = results["boxes"][0].detach().numpy().astype(int)
    cropped_img = pil_img.crop((box[0], box[1], box[2], box[3]))
    cropped_img = cropped_img.resize((224, 224))

    # Підготовка до класифікації
    img_array = keras_image.img_to_array(cropped_img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    pred = clf_model.predict(img_array, verbose=0)
    pred_class = class_names[np.argmax(pred)]

    print(f"{img_file} → {pred_class}")
