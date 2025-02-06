from network import Network

training_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0]),
]

network = Network([2, 2, 1])

network.train(training_data, learning_rate=0.1, epochs=100000)

for inputs, targets in training_data:
    prediction = network.forward(inputs)
    print(f"Input: {inputs}, Prediction: {prediction}")