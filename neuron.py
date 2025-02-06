import random
from utils import sigmoid

class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
        self.output = 0
        self.delta = 0 

    def activate(self, inputs):
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        self.output = sigmoid(weighted_sum)
        
        return self.output