import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# 📐 Налаштування
img_size = (224, 224)
batch_size = 32
data_dir = "data/catvsdog100/"
train_dir = "data/catvsdog100/train"
val_dir = "data/catvsdog100/val"

# 📊 Завантаження даних
# datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input, validation_split=0.2
# )

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

# train_generator = datagen.flow_from_directory(
#     data_dir,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode="categorical",
#     subset="training",
#     shuffle=True,
# )

# val_generator = datagen.flow_from_directory(
#     data_dir,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode="categorical",
#     subset="validation",
# )

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
)

# 📦 Кількість класів
num_classes = train_generator.num_classes
class_names = list(train_generator.class_indices.keys())

# 🧠 Створення моделі (Functional API)
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(num_classes, activation="softmax")(x)

# inputs = tf.keras.Input(shape=(224, 224, 3))
# x = base_model(inputs, training=False)
# x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dense(128, activation="relu")(x)
# x = layers.Dropout(0.3)(x)
# outputs = layers.Dense(num_classes, activation="softmax")(x)

# model = tf.keras.Model(inputs, outputs)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# ⚙️ Компіляція
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 🏋️‍♂️ Навчання
# history = model.fit(train_generator, validation_data=val_generator, epochs=10)


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint("models/base_model.keras", save_best_only=True),
    ],
)

# 💾 Збереження моделі
# os.makedirs("models", exist_ok=True)
# model.save("models/my_model.keras")


# === Розморожування верхніх шарів для fine-tuning ===
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"]
)

print("\n=== Fine-tuning верхніх шарів ===")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    verbose=1,
    callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint("models/fine_tuned_model.keras", save_best_only=True),
    ],
)

print("\n✅ Навчання завершено. Модель збережено у: models/fine_tuned_model.keras")


# 📈 Побудова графіків
def plot_history(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Loss")

    plt.tight_layout()
    plt.savefig("models/training_plot.png")
    plt.show()


plot_history(history)
