import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# 📐 Налаштування
img_size = (224, 224)
batch_size = 32
data_dir = "data/catvsdog100/"

# 📊 Завантаження даних
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
)

# 📦 Кількість класів
num_classes = train_generator.num_classes
class_names = list(train_generator.class_indices.keys())

# 🧠 Створення моделі (Functional API)
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# ⚙️ Компіляція
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 🏋️‍♂️ Навчання
history = model.fit(train_generator, validation_data=val_generator, epochs=10)

# 💾 Збереження моделі
os.makedirs("models", exist_ok=True)
model.save("models/my_model.keras")


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
