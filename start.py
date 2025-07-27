import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

print(tf.config.list_physical_devices('GPU'))

# Завантаження даних
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Нормалізація даних (зводимо значення пікселів до діапазону [0, 1])
x_train, x_test = x_train / 255.0, x_test / 255.0

# Перетворення міток у формат one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Перевірка розмірів даних
print(f"Train data shape: {x_train.shape}, Test data shape: {x_test.shape}")


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 класів для CIFAR-10
])

model.summary()


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test), batch_size=64)


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.2f}")


# Вибір випадкового зображення з тестового набору
idx = np.random.randint(0, x_test.shape[0])
image = x_test[idx]
label = np.argmax(y_test[idx])

# Прогноз класу
prediction = np.argmax(model.predict(image[np.newaxis, ...]))

# Візуалізація результату
plt.imshow(image)
plt.title(f"Actual: {label}, Predicted: {prediction}")
plt.show()


model.save('image_classifier_model.keras')
