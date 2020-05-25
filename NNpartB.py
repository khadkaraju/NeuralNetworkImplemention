# Raju Khadka
# Machine Learning Project 5
# Part B

import numpy as np
import matplotlib.pyplot as plt


# generating datasets
X = 2*np.random.uniform(size=(50, 1))-1
print("Input: ")
print(X)

T = np.sin(2*np.pi*X)+0.3*np.random.uniform(size=(50, 1))
print("Target: ")
print(T)

# N_inputLayers=1
# N_hiddenLayers=3
# N_outputLayers=1


epochs = 3000
rho = 0.01

# Sigmoidal Activation Function and its derivative for each layer


def ActivationFunc(z):
    return np.tanh(z)


def ActivationFunc_D(z):
    return 1-(z)**2

# initializing random weights and bias


def Network(N_inputLayers, N_hiddenLayers, N_outputLayers):
    hidden_W = np.random.uniform(size=(N_inputLayers, N_hiddenLayers))
    output_W = np.random.uniform(size=(N_hiddenLayers, N_outputLayers))

    hidden_B = np.random.uniform(size=(1, N_hiddenLayers))
    output_B = np.random.uniform(size=(1, N_outputLayers))

    cost_func = []

    for value in range(epochs):
        # Forward Network

        # for hidden layer
        hidden_layer = (np.dot(X, hidden_W))+hidden_B
        # applying activation function for output
        hidden_output = ActivationFunc(hidden_layer)

        # for output layer
        output_layer = (np.dot(hidden_output, output_W)) + output_B
        output_P = ActivationFunc(output_layer)  # output prediction

        # Backpropagation
        loss = 2 * (T-output_P)  # derivative of MSE (generating loss)

        output_derivative = loss * ActivationFunc_D(output_P)
        error_hidden_layer = output_derivative.dot(output_W.T)

        hidden_layer_derivative = error_hidden_layer * \
            ActivationFunc_D(hidden_output)

        # Updating Weights and Biases using GD
        # for output layer
        output_W += rho * hidden_output.T.dot(output_derivative)
        output_B += rho * np.sum(output_derivative, axis=0, keepdims=True)

        # for hidden layer
        hidden_W += rho * X.T.dot(hidden_layer_derivative)
        hidden_B += rho * np.sum(hidden_layer_derivative,
                                 axis=0, keepdims=True)

        # generating cost

        cost = np.sum((np.sqrt((T-output_P)**2)))/50
        cost_func.append(cost)
        print(cost)
    return cost_func, output_P


# for 3 hidden units
cost_func, output_P = Network(1, 3, 1)

plt.subplot(2, 2, 1)
plt.plot(cost_func)
plt.xlabel("epochs")
plt.ylabel("Error")
plt.title("For 3 hidden units => loss vs epochs")


plt.subplot(2, 2, 2)
plt.scatter(X, output_P, label="Output Prediction", marker="o", color="blue")
plt.scatter(X, T, label="Target", marker="o", color="red")
plt.xlabel('input (X)')
plt.ylabel('Predicted Output and Target')
plt.title("Predicted Output and Target Vs Input (X)")
plt.legend()


# for 20 hidden units
cost_function, output_Pre = Network(1, 20, 1)

plt.subplot(2, 2, 3)
plt.plot(cost_function)
plt.xlabel("epochs")
plt.ylabel("Error")
plt.title(" For 20 hidden units => loss vs epochs")


plt.subplot(2, 2, 4)
plt.scatter(X, output_Pre, label="Output Prediction", marker="o", color="blue")
plt.scatter(X, T, label="Target", marker="o", color="red")
plt.xlabel('input (X)')
plt.ylabel('Predicted Output and Target')
plt.title("Predicted Output and Target Vs Input (X)")

plt.tight_layout()
plt.legend()
plt.show()
