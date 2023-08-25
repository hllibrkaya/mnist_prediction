# NN MNIST Digit Recognition Project

This project is developed using the Python programming language to solve the digit recognition problem on the MNIST dataset using  Neural Networks (NN).

## Project Description

In this project, the MNIST dataset is used as the dataset. The MNIST dataset consists of hand-written digits in a 28x28 pixel format. The goal is to accurately recognize a given hand-written digit.

## Artificial Neural Network Features

The artificial neural network used in this project has the following features:


- ## Configurable Neuron Counts

While the artificial neural network used in this project has a certain architecture for its layers and neuron counts, it can be customized according to your needs. By editing the project code, you can modify the neuron counts in the input, hidden, and output layers. This flexibility allows you to tailor the network for different datasets or problems.

For instance:

- Input Layer: You can adjust the neuron count based on the data dimension.
- Hidden Layer: You can increase or decrease the neuron count based on different complexity levels.
- Output Layer: You can modify the neuron count according to the categories you are classifying.

This customizability provides you with the flexibility to adapt your project to different datasets or problems.

---

The neuron counts and layer structures of the project can influence learning capabilities. By experimenting with different architectures and observing the results, you can achieve the best outcomes.


### Learning Rate

The learning rate of the artificial neural network is set to 0.3.

### Weight Matrices

The weight matrices of the artificial neural network are initialized with random values. Weight matrices represent the connections between the hidden layer and the output layer.

### Activation Function

The activation function of the neurons in the network is set to the sigmoid function.

### Training and Testing

The project is trained and tested on the MNIST dataset. In the training step, the network updates its weights using the provided data to learn. In the testing step, the trained network evaluates accuracy on new data.

### Accuracy Measurement

The project measures accuracy on test data. Predicted results are compared with actual labels, and the percentage of correct predictions is calculated.

## Requirements

To run this project, the following requirements are necessary:
- Python (3.x recommended)
- NumPy library
- Matplotlib library
- SciPy library
- tqdm library


