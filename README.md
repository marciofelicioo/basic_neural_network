# Neural Network for Digit Recognition "0" and "1"

This project is a Java implementation of a neural network designed to recognize handwritten digits, specifically distinguishing between digits "0" and "1", using a simplified version of the MNIST dataset. The images are grayscale with a resolution of 20x20 pixels, resulting in 400 input features.

The project is an adaptation and extension of the basic neural network library provided by [kim-marcel/basic_neural_network](https://github.com/kim-marcel/basic_neural_network), with modifications to meet the specific requirements of this digit recognition task.

## Features

### Customizable Neural Network Architecture
- **400 input neurons** corresponding to the 400 pixels of the 20x20 images.
- **Configurable hidden layer(s):** One hidden layer with 10 neurons, balancing performance and computational efficiency.
- **1 output neuron** providing the probability that the digit is "1".

### Functionality
- **Sigmoid Activation Function** for all neurons, ensuring outputs in the range [0, 1].
- **Adjustable Learning Rate** for training optimization.
- **Early Stopping Mechanism** to prevent overfitting by stopping training when appropriate.

### Data Preprocessing
- **Normalization** of input data to the range [0, 1].
- **Data integrity validation,** ensuring each input has 400 pixels and binary labels.

### Modular Design with Clear Separation of Components
- `DataPreprocessor`: Loading and preprocessing of data.
- `DigitTrainer`: Training and validation of the neural network.
- `DigitClassifier`: Classification of new inputs using the trained network.
- `NeuralNetwork`: Core implementation of the neural network.

### Additional Features
- **Unit Tests** to ensure reliability and correctness of each component.
- **Saving and Loading Weights** of the neural network for model reuse.

## Getting Started

### Prerequisites
- **Java Development Kit (JDK) 8** or higher.
- **Maven** for dependency management.

### Installing
Clone the repository:
git clone git@github.com:marciofelicioo/basic_neural_network
