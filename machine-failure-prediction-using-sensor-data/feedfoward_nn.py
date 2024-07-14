import math, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

"""
Feedfoward Neural Network Model (FFNN).
Supervised Learning.
Binary Classification.
"""

# Read the dataset
original_df = pd.read_csv('data/data.csv')

"""
Show histograms (probability distribution) of passed data for all features.
Separates data into machine failure and no machine failure.
"""
def show_histograms(df):
    for i in range(len(df.columns[:-1])):
        label = df.columns[i]
        plt.hist(df[df['fail']==1][label], color='red', label='Machine Failure', alpha=0.7, density=True, bins=amount_of_bins)
        plt.hist(df[df['fail']==0][label], color='green', label='No Machine Failure', alpha=0.7, density=True, bins=amount_of_bins)
        plt.title(label)
        plt.ylabel('Probability')
        plt.xlabel(label)
        plt.legend()
        plt.show()


# Count of machine failure or not machine failure
original_machine_fail_count = len(original_df[original_df['fail']==1])
original_machine_not_fail_count = len(original_df[original_df['fail']==0])

# Formula using Square-Root Choice
amount_of_bins = int(math.ceil(math.sqrt(len(original_df))))

# show_histograms(original_df)

features = original_df[original_df.columns[:-1]].values
labels = original_df[original_df.columns[-1]].values

# Scaling the data
scaler = StandardScaler()
features = scaler.fit_transform(features)
"""
 Reshaping the data.
 Since labels are a 1D matrix while features are a 2D matrix
 New format will be (original_value, 1)
 """
data = np.hstack((features, np.reshape(labels, (-1, 1))))
transformed_df = pd.DataFrame(data, columns=original_df.columns)
# show_histograms(transformed_df)

"""
Oversampling
As the size of the data is different, we need to oversample it
"""
oversample = RandomOverSampler()
features, labels = oversample.fit_resample(features, labels)
data = np.hstack((features, np.reshape(labels, (-1, 1))))
transformed_df = pd.DataFrame(data, columns=original_df.columns)
# show_histograms(transformed_df)

# Count of machine failure == count of no machine failure
resampled_machine_fail_count = len(transformed_df[transformed_df['fail']==1])
resampled_machine_not_fail_count = len(transformed_df[transformed_df['fail']==0])

# Split data into train and test
features_train, features_temporary, labels_train, labels_temporary = train_test_split(features, labels, train_size=0.4, random_state=0)
features_validation, features_test, labels_validation, labels_test = train_test_split(features_temporary, labels_temporary, train_size=0.5, random_state=0)

# Model creation
current_neurons = 16
model = tf.keras.Sequential([
    # Dense layers with 16 neurons each
    # Relu activation: If x <= 0, return 0; else, return x.
    tf.keras.layers.Dense(current_neurons, activation='relu'),
    tf.keras.layers.Dense(current_neurons, activation='relu'),
    # Output layer with 1 neuron with sigmoid activation
    # Sigmoid activation: returns either 0 or 1. (Binary classification)
    tf.keras.layers.Dense(1, activation='sigmoid')
])

"""
Compile the model.
Given that we are perfoming binary classification knowing the labels, we will use the BinaryCrossentropy loss function.
"""
current_learning_rate = 0.001
current_optimizer = tf.keras.optimizers.Adam(learning_rate=current_learning_rate)
current_loss_function = tf.keras.losses.BinaryCrossentropy()
current_metrics = tf.keras.metrics.Accuracy()
model.compile(optimizer = current_optimizer, 
              loss = current_loss_function, 
              metrics = [current_metrics])

# Train the model
current_epochs = 20
current_batch_size = 16
model.fit(features_train, labels_train, epochs = current_epochs, batch_size = current_batch_size, validation_data = (features_validation, labels_validation))

# Save the model
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
model.save(f'trained_models/model_{current_datetime}.keras')
