import numpy as np
from numpy.random import default_rng
import data_gen as dg
import json

def softmax(x):
    max = np.max(x)
    dim = len(x)
    res = np.exp(x-max) / np.sum(np.exp(x[i]-max) for i in range(dim))
    return res

class NeuronalNetwork():
    def __init__(self, sizes:list, weights:np.ndarray=None, biases:np.ndarray=None, non_linear_funktion:str="sigmoid", name="Brian"):
        """``sizes`` = [2,10,10,2] for creating a NN with In/Output of size 2, and hidden layers of size 10.
        \nWeights and biases are set random if not specified further.
        \nSelect one of the following non-linear funktions for neuron simulation: ``sigmoid`` (default), ``ReLU``, ``tanh``"""
        rng = default_rng()
        self.name = name
        self.sizes:list = sizes
        self.input_size:int = self.sizes[0]
        self.sizes.__delitem__(0)
        self.L = len(self.sizes)
        self.nlf:str = non_linear_funktion
        self.z = []

        if weights == None:
            self.weights = [rng.random((self.sizes[0], self.input_size))]
            for l in range(self.L-1):
                self.weights.append(rng.random((self.sizes[l+1], self.sizes[l])))
        else:
            self.weights = weights
        
        if biases == None:
            self.biases = [rng.random((self.sizes[l])) for l in range(self.L)]
        else:
            self.biases = biases
        
    def forward_prop(self, x) -> np.ndarray:
        self.z = [x]
        a = x
        for w,b in zip(self.weights, self.biases):
            self.z.append(np.dot(w, a) + b)
            a = self.sigma(self.z[-1])
        return a
    
    def back_prop(self, guess, y, lr):
        delta = guess - y
        for l in range(self.L-1, -1, -1):
            delta = delta * self.sigma_prime(self.z[l+1])

            self.biases[l] -= lr * delta
            self.weights[l] -= lr * np.outer(delta, self.z[l])

            delta = np.dot(self.weights[l].T, delta)

# Add batch-learning and random data select
    def train_on(self, data=list, iter:int=100, learning_rate:float=0.01): 
        """``data`` has to be a list of tupels like ``[(x0, y0), (x1, y1), ...]`` \
            with ``y`` beeing the expected output to the input ``x``.
            Both ``x`` and ``y`` must have the dimesion specified on creation of the nn."""
        for i in range(iter):
            print("Epoch",i+1,"of",iter)
            for x,y in data:
                x = np.array(x)
                y = np.array(y)
                guess = self.forward_prop(x)
                self.back_prop(guess, y, learning_rate)

    def sigma(self, x) -> np.ndarray:
        match self.nlf:
            case "sigmoid":
                return 1/(1+np.exp(-x))
            case "ReLU":
                return np.maximum(0,x)
            case "tanh":
                return np.tanh(x)
            
    def sigma_prime(self, x) -> np.ndarray:
        match self.nlf:
            case "sigmoid":
                return np.exp(-x)/((1+np.exp(-x))**2)
            case "ReLU":
                return np.where(x > 0, 1, 0)
            case "tanh":
                return 1 - np.tanh(x) * np.tanh(x)

    def print_parameters(self):
        print("Weights:\n")
        for l in range(self.L-1):
            print("Layer", l, ":\n", self.weights[l])
        print("\nBiases:\n")
        for l in range(self.L-1):
            print("Layer", l, ":\n", self.biases[l])

nn = NeuronalNetwork([9,3,5,5,3,9], non_linear_funktion="tanh")

print("\nGuess befor training:", nn.forward_prop([1,2,0,1,0,0,0,0,0]))

try:
    with open('data.json', 'r') as f:
        data = json.load(f)
except: 
    data_new = dg.generate_dataset(10000)
    with open('data.json', 'w') as f:
        json.dump(data_new, f)

nn.train_on(data, 100, 0.01)

guess = nn.forward_prop([1,2,0,1,0,0,0,0,0])
print("\nGuess after training:", guess)
print("\nGuess after training:", np.max(guess))