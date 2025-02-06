from layer import Layer
from utils import sigmoid_derivative

class Network:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i + 1], layer_sizes[i])
            self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def train(self, training_data, learning_rate=0.1, epochs=1000):
        for epoch in range(epochs):
            for inputs, targets in training_data:
                outputs = self.forward(inputs)

                for i in reversed(range(len(self.layers))):
                    layer = self.layers[i]
                    errors = []
                    if i == len(self.layers) - 1:
                        for j in range(len(layer.neurons)):
                            neuron = layer.neurons[j]
                            errors.append(targets[j] - outputs[j])
                    else:
                        for j in range(len(layer.neurons)):
                            error = sum(self.layers[i + 1].neurons[k].delta * self.layers[i + 1].neurons[k].weights[j] for k in range(len(self.layers[i + 1].neurons)))
                            errors.append(error)

                    for j in range(len(layer.neurons)):
                        neuron = layer.neurons[j]
                        neuron.delta = errors[j] * sigmoid_derivative(neuron.output)
                        for k in range(len(neuron.weights)):
                            neuron.weights[k] += learning_rate * neuron.delta * (inputs[k] if i == 0 else self.layers[i - 1].neurons[k].output)
                        neuron.bias += learning_rate * neuron.delta