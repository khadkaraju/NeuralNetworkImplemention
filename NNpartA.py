# Raju Khadka
# Machine Learning Project 5
# Part A

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# datasets
inputs = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]])
print("Input: ")
print(inputs)

target = np.array([[0], [0], [1], [1]])
print("Target: ")
print(target)


# C1=[1,0]
# C2=[0,1]

# expected_output = []
# for i in inputs:
# 	temp_T=[]
# 	if i[0]==i[1]:
# 		expected_output.append(C1)
# 	else:
# 		expected_output.append(C2)
# 	# expected_output.append(temp_T)
# expected_output=np.array(expected_output)

# print(expected_output)


N_inputLayers = 2
N_hiddenLayers = 2
N_outputLayers = 1

# initializing number of epochs and learning rate
epochs = 5000
rho = 0.1

# Sigmoidal Activation Function and its derivative for each layer


def ActivationFunc(z):
    return 1/(1 + np.exp(-z))


def ActivationFunc_D(z):
    return z * (1 - z)


# initializing random weights and bias
hidden_W = np.random.uniform(size=(N_inputLayers, N_hiddenLayers))
output_W = np.random.uniform(size=(N_hiddenLayers, N_outputLayers))

hidden_B = np.random.uniform(size=(1, N_hiddenLayers))
output_B = np.random.uniform(size=(1, N_outputLayers))


def ForwardNetwork(inputs):
    # Forward Network

    # for hidden layer
    hidden_layer = (np.dot(inputs, hidden_W))+hidden_B
    # applying activation function for output
    hidden_output = ActivationFunc(hidden_layer)

    # for output layer
    output_layer = (np.dot(hidden_output, output_W)) + output_B
    output_P = ActivationFunc(output_layer)  # output prediction

    return output_P, hidden_output


def BackProp(output_P, hidden_output):
    # Backpropagation
    loss = target - output_P  # generating loss

    output_derivative = loss * ActivationFunc_D(output_P)
    hidden_layer_loss = output_derivative.dot(output_W.T)

    hidden_layer_derivative = hidden_layer_loss * \
        ActivationFunc_D(hidden_output)
    return hidden_output, output_derivative, hidden_layer_derivative


cost_func = []
counter = 1

for value in range(epochs):

    output_P, hidden_output = ForwardNetwork(inputs)
    hidden_output, output_derivative, hidden_layer_derivative = BackProp(
        output_P, hidden_output)

    # Updating Weights and Biases using GD
    # for output layer
    output_W += rho * hidden_output.T.dot(output_derivative)
    output_B += rho * np.sum(output_derivative, axis=0, keepdims=True)

    # for hidden layer
    hidden_W += rho * inputs.T.dot(hidden_layer_derivative)
    hidden_B += rho * np.sum(hidden_layer_derivative, axis=0, keepdims=True)

    # generating cost
    cost = -np.sum(target*(np.log(output_P)))
    cost_func.append(cost)

    print(cost)


plt.plot(cost_func)
plt.xlabel("epochs")
plt.ylabel("error")
plt.title("error vs epochs")

xx, yy = np.meshgrid(np.arange(-1, 1, 0.1), np.arange(-1, 1, 0.1))
y_pred, _ = ForwardNetwork(np.array([xx.ravel(), yy.ravel()]).T)
y_pred = y_pred.reshape(xx.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X=xx, Y=yy, Z=y_pred)
plt.show()
