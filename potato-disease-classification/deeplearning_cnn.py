from enum import Enum
import os, random, datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras

"""
Deep Learning Convolutional Neural Network.
Supervised Learning.
Image Classification.

Models are saved with the following naming convention:
model_{epochs}_{batch_size}_{validation_split}_{datetime}.keras
"""

DATADIR = "data"
CATEGORIES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
class ChannelDimensions(Enum):
    GRAY = 1
    RGB = 3

# Flag to indicate which operations to run based on the channel dimension.
current_channel = ChannelDimensions.RGB.value

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        # Grayscale
        if current_channel == 1: 
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            plt.imshow(img_array, cmap="gray")
            break
        # RGB
        elif current_channel == 3:
            img_array = cv2.imread(os.path.join(path, img))
            plt.imshow(img_array)
            break
    break

IMG_SIZE = 256
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
# Grayscale
if current_channel == 1: 
    plt.imshow(new_array, cmap="gray")
# RGB
elif current_channel == 3:
    plt.imshow(new_array)

# Training Data
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                # Grayscale
                if current_channel == 1: 
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                # RGB
                elif current_channel == 3:
                    img_array = cv2.imread(os.path.join(path, img))

                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
        break

create_training_data()

random.shuffle(training_data)

X = [] # Features
y = [] # Labels

for features, label in training_data:
    X.append(features)
    y.append(label)

# Format of the input data for the model (batch_size, height, width, channels)
# Grayscale
if current_channel == 1:
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, ChannelDimensions.GRAY.value)
# RGB
elif current_channel == 3:
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, ChannelDimensions.RGB.value)

y = np.array(y)

# Convert values to decimal in range 0-1
X = X/255

# Model creation
conv_filters = 64
# Grayscale
if current_channel == 1:
    model = keras.models.Sequential([
        # Convulational layer 1
        keras.layers.Conv2D(filters=conv_filters, kernel_size=(3, 3), activation="relu", data_format="channels_last"),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        # Convulational layer 2
        keras.layers.Conv2D(filters=conv_filters, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        # Flattening
        keras.layers.Flatten(),
        # Hidden layer
        keras.layers.Dense(units=64, activation="relu"),
        # Output layer (3 categories)
        keras.layers.Dense(units=3, activation="softmax")
    ])

# RGB
elif current_channel == 3:
    model = keras.models.Sequential([
        # Convulational layer. Given that RGB images have more data, additional layers are not needed.
        keras.layers.Conv2D(filters=conv_filters, kernel_size=(3, 3), activation="relu", data_format="channels_last"),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        # Flattening
        keras.layers.Flatten(),
        # Hidden layer
        keras.layers.Dense(units=64, activation="relu"),
        # Output layer (3 categories)
        keras.layers.Dense(units=3, activation="softmax")
    ])


"""
Compile the model.
Given that we are perfoming categorical classification knowing the labels, we will use the SparseCategoricalCrossentropy loss function.
"""
current_learning_rate = 0.001
current_optimizer = keras.optimizers.Adam(learning_rate=current_learning_rate)
current_loss_function = keras.losses.SparseCategoricalCrossentropy()
current_metrics = tf.keras.metrics.SparseCategoricalAccuracy()
model.compile(optimizer=current_optimizer, loss=current_loss_function, metrics=[current_metrics])


# Train the model
current_epochs = 5
current_batch_size = 32
current_validation_split = 0.1
model.fit(X, y, epochs=current_epochs, batch_size=current_batch_size, validation_split=current_validation_split)


# Save the model
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
if current_channel == 1:
    model.save(f'trained_models/model_grayscale_{current_epochs}_{current_batch_size}_{current_validation_split}_{current_datetime}.keras') 
elif current_channel == 3: 
    model.save(f'trained_models/model_rgb_{current_epochs}_{current_batch_size}_{current_validation_split}_{current_datetime}.keras')
else:
    print("Invalid channel")
