# ----------------------------------------------------------------------------------------------------------------------
# NOTE: excessive in-line comments were just for my own learning purposes
# ----------------------------------------------------------------------------------------------------------------------

# Importing necessary libraries
import torch
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
x_training = torch.tensor(range(1, 1000), dtype=torch.float32).view(-1, 1)                  # Celsius temperatures Nx1 column vector
y_training = torch.tensor(list(map(celsius_to_fahrenheit, x_training)), dtype=torch.float32).view(-1, 1)  # Fahrenheit temperatures Nx1 column vector

# Creating dataset
dataset = torch.utils.data.TensorDataset(x_training, y_training)

# Loading dataset (batching is for processing efficiency, shuffling is to avoid recurring local minima)
dataset = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Creating neural network
neural_network = torch.nn.Sequential(torch.nn.Linear(1, 1))                                 # Single-layer, single-input, single-output fully-connected neural network (celsius in, fahrenheit out)

# Defining loss (error) function
loss_function = torch.nn.MSELoss()                                                          # Quantifying difference between predicted label and actual label  (does not need normalized softmax y_pred input)

# Defining parameter updating algorithm
optimizer = torch.optim.Adam(neural_network.parameters(), lr=LEARNING_RATE)                 # Adam is alternative to classic gradient descent

# Defining empty containers to track losses over training
loss_history = []

# Training the neural network
for epoch in tqdm(range(EPOCH_NUMBER)):                                                     # For each pass over the full data set (tqdm for runtime progression bar display)
    loss = 0                                                                                # Resetting loss for current epoch
    for x, y in dataset:                                                                    # For each batch of samples in data set
        optimizer.zero_grad()                                                               # Clearing gradient since backward() accumulates gradient already
        y_predicted = neural_network(x)                                                     # Computing predictions of current model
        loss = loss_function(y_predicted, y)                                                # Finding loss between predictions and actual labels
        loss.backward()                                                                     # Computing the gradient of the loss WRT each model parameter
        optimizer.step()                                                                    # Updating model parameters at learning rate according to direction of steepest descent
    loss_history.append(loss)                                                               # Tracking accumulated loss over current epoch

# Displaying final neural network parameters
for parameter in neural_network.parameters():
    print(parameter.data)

# Plotting training losses
plt.figure(figsize=(9, 3))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.xlabel('EPOCH')
plt.ylabel("LOSS")

# Generating test data
x_testing = torch.tensor(range(-100, 100), dtype=torch.float32).view(-1, 1)                 # Celsius temperatures Nx1 column vector
y_testing = torch.tensor(list(map(celsius_to_fahrenheit, x_testing)), dtype=torch.float32).view(-1, 1)  # Fahrenheit temperatures Nx1 column vector

# Testing test data
y_testing_predicted = neural_network(x_testing)

# Plotting testing results
plt.subplot(1, 2, 2)
plt.plot(x_testing.numpy(), y_testing.numpy())                                              # Label line
plt.plot(x_testing.numpy(), y_testing_predicted.detach().numpy())                           # Prediction line
plt.xlabel('C')
plt.ylabel("F")
plt.legend(['True Value', 'Predicted Value'])
plt.show()
