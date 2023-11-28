## Neural Networks
The main feature of this repository contains code for evaluating Hessian-vector products for neural networks using matrix differential calculus and dynamical systems theory. Hessian-vector products are useful for network pruning, optimal data collection, and measuring the sensitivity of the network to data removal, to name a few applications. 

<p align="center">
  <img src="https://github.com/wbbreslin/NeuralNetwork/blob/main/Images/NNet1.png">
</p>

Calculating the gradient requires a forward pass through the network, and a backward pass through an adjoint model (first-order adjoint). Computing Hessian-vector products involves another forward pass through a tangent-linear model (TLM) and another backward pass through the second-order adjoint model, as pictured in the diagram below. 

<p align="center">
  <img src="https://github.com/wbbreslin/NeuralNetwork/blob/main/Images/SOA.png" width=75% height=75%>
</p>

## How to Use
Download the .py files in the Neural Network Library folder. There are four files:
* ```Base.py``` - this contains supplemental functions that are not directly related to the NN model
* ```FirstOrderModel.py``` - contains the functions to evaluate the gradient for a NN
* ```SecondOrderModel.py``` - contains the functions to evalute the Hessian-vector product for a NN
* ```TrainingAlgorithms.py``` - contains optimization algorithms (e.g. gradient descent)

```FirstOrderModel.py``` and ```TrainingAlgorithms.py``` together make up the usual backpropagation algorithm.

While the ```SecondOrderModel.py``` file can be used for NN training, due to the computational complexity, its primary uses are for network pruning and measuring observation sensitivities of various aspects of the forecast / predictive model.

To apply these models, data should be contained in an (n x p) matrix, where the n rows each contain one data point in p-variables. The setup is the same as the design matrix in linear regression.

## Project Files
The project files call the functions in the .py files mentioned above, and apply them to various data sets. Each of the project files are independent projects that I am working on as a part of my dissertation.

## Example of Training a Neural Network
The code below comes from the ```SIAM2019.py``` file in the projects folder, which is a simple example for training a neural network.

First, we import the necessary modules.
```{python}
import numpy as np
import Base as base
import TrainingAlgorithms as train
```

The dataset:

```{python}
"""The data set of predictor variables"""
x_predictors = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
                        [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]]).T

"""The data set of outcome variables"""
y_outcomes = np.array([[1,1,1,1,1,0,0,0,0,0],
                      [0,0,0,0,0,1,1,1,1,1]]).T
```
The X data is 2-dimensional, representing a point in space, and the Y data is a binary outcome for each of the X values (a success or failure).

Then, we need to define the network structure, which we can do using the ```create_network(neurons)``` function from Base.py. We also define a list of the activation function types for the network. The ```augment_network(weights, biases)``` combines the bias terms into the weight matrix.

```{python}
neurons = np.array([2,2,3,2])
activations = ["sigmoid","sigmoid","sigmoid","sigmoid"]
weights, biases = base.create_network(neurons)
augmented_weights, constants = base.augment_network(weights, biases)
```
Create a neural network object as a dictionary.
```{python}
nnet = {'Predictors': x_predictors,
        'Outcomes': y_outcomes,
        'Weights': weights,
        'Neurons': neurons}
```
Now train the network by calling the optimization algorithms from ```TrainingAlgorithms.py```.
```{python}
"""Train the neural network"""
nnet = train.gradient_descent(nnet,max_iterations=10**4)
```
Various aspects of the model can be accessed through the following dictionary keys.
```{python}
dict_keys([ 'Predictors',
            'Outcomes',
            'Weights',
            'Neurons',
            'States',
            'Augmented_States',
            'Augmented_Weights',
            'First_Derivatives',
            'Lambdas',
            'Gradients'])
```
Here is a visualization of the trained NN. 

<p align="center">
  <img src="https://github.com/wbbreslin/NeuralNetwork/blob/main/Images/Region.png" width=50% height=50%>
</p>

This was trained using gradient descent on the full data set (n=10). Below shows the convergence of the cost function, for both the gradient decent algorithm, and the stochastic gradient method using half-data batches (n=5).

<p align="center">
  <img src="https://github.com/wbbreslin/NeuralNetwork/blob/main/Images/CostFunctions.png" width=50% height=50%>
</p>

The code for training this model and making these plots can be found in the Highams2019 folder.

## Calculating Hessian-Vector Products
To calculate the Hessian-Vector product, we do a forward and backward pass through the first-order model, then do another forward and backward pass using the second-order model. For many applications, we start with an already trained neural network, and do another forward and backward pass in the second-order model.

First, import the second-order model. 
```{python}
import SecondOrderModel as som
```
If you haven't already trained the network, you will either need to train it first, or import the first-order model, and run that before the second-order model can be ran.

Then create some vectors to use for the Hessian-vector products. For simplicity, here we just use the gradients. These start as matrices, so these need to be converted to vectors.

```{python}
vectors = nnet['Gradients'].copy()
for i in range(len(vectors)):
    vectors[i], dims = base.to_vector(vectors[i])
```

With the chosen vectors, we can now do the forward and backward passes in the second-order model.
```{python}
nnet = som.forward_pass(nnet,vectors)
nnet = som.backward_pass(nnet, vectors)
```
The second-order model adds the following keys to ```nnet```.
```{python}
['Thetas', 'Second_Derivatives', 'Omegas', 'Hv_Products']
```

By passsing through every possible unit vector, the full Hessian matrix can be constructed. Below is a heatmap of the Hessian matrix after training for 4000 iterations.
<p align="center">
  <img src="https://github.com/wbbreslin/NeuralNetwork/blob/main/Images/Hessian.png" width=50% height=50%>
</p>
