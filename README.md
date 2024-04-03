## Neural Networks
The main feature of this repository contains code for evaluating Hessian-vector products for neural networks using matrix differential calculus and dynamical systems theory. Hessian-vector products are useful for network pruning, optimal data collection, and measuring the sensitivity of the network to data removal, to name a few applications. 

<p align="center">
  <img src="https://github.com/wbbreslin/NeuralNetwork/blob/main/Images/NNet1.png">
</p>

Calculating the gradient requires a forward pass through the network, and a backward pass through an adjoint model (first-order adjoint). Computing Hessian-vector products involves another forward pass through a tangent-linear model (TLM) and another backward pass through the second-order adjoint model, as pictured in the diagram below. 

<p align="center">
  <img src="https://github.com/wbbreslin/NeuralNetwork/blob/main/Images/SOA.png" width=50% height=50%>
</p>

## How to Use
Two example files are provided to demonstrate how to use this package to train a neural network. Additional files will be uploaded to demonstrate the Hessian-vector products and their applications at a later date.

## Images

Here is a visualization of the trained NN. 

<p align="center">
  <img src="https://github.com/wbbreslin/NeuralNetwork/blob/main/Images/Region.png" width=50% height=50%>
</p>

This was trained using gradient descent on the full data set (n=10). Below shows the convergence of the cost function, for both the gradient decent algorithm, and the stochastic gradient method using half-data batches (n=5).

<p align="center">
  <img src="https://github.com/wbbreslin/NeuralNetwork/blob/main/Images/CostFunctions.png" width=75% height=75%>
</p>

By passsing through every possible unit vector, the full Hessian matrix can be constructed. Below is a heatmap of the Hessian matrix after training for 4000 iterations.
<p align="center">
  <img src="https://github.com/wbbreslin/NeuralNetwork/blob/main/Images/Hessian.png" width=50% height=50%>
</p>
