# ----------------------------------------------------------------------------------------------------------------------
# NOTE: excessive in-line comments were just for my own learning purposes
# ----------------------------------------------------------------------------------------------------------------------

# Importing necessary libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import tqdm                                                                       # For progress bar
import matplotlib.pyplot as plt                                                             # For plotting


# Defining temperature conversion function
def celsius_to_fahrenheit(celsius):
    fahrenheit = (1.8 * celsius) + 32
    return fahrenheit


# Defining training constants
LEARNING_RATE = 0.1                                                                         # Step size for gradient descent
EPOCH_NUMBER = 100                                                                          # Number of passes over entire training dataset
BATCH_SIZE = 10                                                                             # Number of samples loaded per batch (for performance enhancement)

# Generating training data
x_training = np.array(list(range(1, 1000))).reshape((-1, 1))                                # Celsius temperatures Nx1 column vector
y_training = np.array(list(map(celsius_to_fahrenheit, x_training))).reshape((-1, 1))        # Fahrenheit temperatures Nx1 column vector

# Creating dataset (batching is for processing efficiency, shuffling is to avoid recurring local minima)
dataset = tf.data.Dataset.from_tensor_slices((x_training, y_training)).shuffle(BATCH_SIZE).batch(BATCH_SIZE)  # Shuffle to avoid recurring local minima

# Creating neural network
neural_network = keras.Sequential()                                                         # Empty sequential neural network (no layers yet)
neural_network.add(keras.layers.Dense(1, input_shape=[1]))                                  # Adding single-layer, single-input, single-output fully-connected layer (celsius in, fahrenheit out)

# Defining loss (error) function
loss_function = tf.keras.losses.MSE                                                         # Quantifying difference between predicted label and actual label (needs normalized softmax y_pred input)

# Defining parameter updating algorithm
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)                           # Adam is alternative to classic gradient descent

# Defining training evaluation metric
accumulated_loss = tf.keras.metrics.Mean(name='loss')

# Defining empty containers to track losses over training
loss_history = []

# Training the neural network
for epoch in tqdm(range(EPOCH_NUMBER)):                                                     # For each pass over the full data set (tqdm for runtime progression bar display)
    accumulated_loss.reset_states()                                                         # Resetting loss for current epoch
    for x, y in dataset:                                                                    # For each batch of samples in data set
        with tf.GradientTape() as grad:
            y_predicted = neural_network(x)                                                 # Computing predictions of current model
            loss = loss_function(y, y_predicted)                                            # Finding loss between predictions and actual labels
            gradients = grad.gradient(loss, neural_network.trainable_variables)             # Computing the gradient of the loss WRT each model parameter
        optimizer.apply_gradients(zip(gradients, neural_network.trainable_variables))       # Updating model parameters at learning rate according to direction of steepest descent
        accumulated_loss(loss)                                                              # Accumulating loss between predictions and actual labels
    loss_history.append(accumulated_loss.result())                                          # Tracking total accumulated loss over current epoch

# Displaying final neural network parameters
for parameter in neural_network.trainable_variables:
    print(parameter.value())

# Plotting training losses
plt.figure(figsize=(9, 3))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.xlabel('EPOCH')
plt.ylabel("LOSS")

# Generating test data
x_testing = np.array(list(range(-100, 100))).reshape((-1, 1))                               # Celsius temperatures Nx1 column vector
y_testing = np.array(list(map(celsius_to_fahrenheit, x_testing))).reshape((-1, 1))          # Fahrenheit temperatures Nx1 column vector

# Testing test data
y_testing_predicted = neural_network(x_testing)

# Plotting testing results
plt.subplot(1, 2, 2)
plt.plot(x_testing, list(y_testing))                                                        # Label line
plt.plot(x_testing, y_testing_predicted)                                                    # Prediction line
plt.xlabel('C')
plt.ylabel("F")
plt.legend(['True Value', 'Predicted Value'])
plt.show()
