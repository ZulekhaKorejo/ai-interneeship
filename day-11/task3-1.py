#back propagation is the process of identifying what caused the error
# and guiding the optimizer how to fix it, it comes after front propagation is done
# it's used to make the neural network learn from their mistakes by telling it how far off it is

import numpy as np
import matplotlib.pyplot as plt

# function for sigmoid to call it later easily, it makes the result between 0 and 1
def sigmoid(x):
    return 1/(1+np.exp(-x))

#sigmoid derivative, it measures the error
def gradient(x):
    return x*(1-x)

#input array where it shows diff cases for xor
x = np.array([[1,1],
             [1,0],
             [0,1],
             [0,0]])

#output array showing what each input resulted in, to train the model
y = np.array([[0], [1], [1], [0]])

# this for stable outcome in each run, it gives the starting point
np.random.seed(42)
#initailize the weights
weight0 = (2*np.random.random((2, 4))-1)
weight1 = (2*np.random.random((4,1))-1)

# a loop for propagation
# in each turn it learns better and adjusts the weight at the end of the cylce
# that's y we didn't define the weights here, it'll make it lose all the progress
#i needed a huge range to make it learn cause the LR is so low as requested
count = 0
loss_curve = []
for epoch in range(10000):
    learning_rate = 0.1

    #layer0 is input x, layer1 is hidden layer, layer2 is output
    layer_0 = x
    layer_1 = sigmoid(np.dot(layer_0,weight0))
    layer_2 = sigmoid(np.dot(layer_1, weight1))

    #this calculates the diff between the ideal result and ours and defining mean squared error for result
    error = y - layer_2
    mse = np.mean(error**2)

    #optimizing
    output_mistakes = error*gradient(layer_2)
    # this calculates how much error was caused by each neuron in the hidden layer
    hidden_mistakes = np.dot(output_mistakes, weight1.T)
    layer1_mistakes = hidden_mistakes*gradient(layer_1)
    
    #gradually making the weights better
    weight1 += np.dot(layer_1.T ,output_mistakes) * learning_rate
    weight0 += np.dot(layer_0.T,layer1_mistakes )*learning_rate

    #collecting mse over the loop for curve loss graph
    loss_curve.append(mse)
    if epoch % 500 == 0:
        count += 500
        print(f'Mean squared error in round {count}: \n{(mse):.4f}\n')

# now we represent that loss b4 showing what it resulted in at the end
plt.Figure(figsize=(18,6))
plt.plot(loss_curve, color = 'skyblue')
plt.title('how the errors reduced over training (loss curve):')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.show()

#layer 2 is the output, y is the ideal answer, they're so close to y so it's correct
print("result:", layer_2)