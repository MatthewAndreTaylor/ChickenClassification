import tensorflow as tf
from tensorflow import keras

# Load the dataset of chicken and non-chicken images
train_data = keras.preprocessing.image_dataset_from_directory(
    'Training/Chickens',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(180, 180),
    batch_size=32)

val_data = keras.preprocessing.image_dataset_from_directory(
    'Training/Chickens',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(180, 180),
    batch_size=32)


class_names = train_data.class_names
print(class_names)
num_classes = len(class_names)

model = keras.models.Sequential([
    keras.layers.Rescaling(1. / 255, input_shape=(180, 180, 3)),
    keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_data, validation_data=val_data, epochs=15)
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    'Tests/Chickens',
    image_size=(180, 180))

test_loss, test_acc = model.evaluate(test_data)
print('Test accuracy:', test_acc)
