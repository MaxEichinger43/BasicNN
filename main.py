import numpy as np
from numpy.random import default_rng
import data_gen as dg
import json

def softmax(x):
    max_val = np.max(x)
    exp_x = np.exp(x - max_val)
    return exp_x / np.sum(exp_x)

class NeuronalNetwork:
    def __init__(self, sizes: list, weights: np.ndarray = None, biases: np.ndarray = None, 
                 non_linear_funktion: str = "sigmoid", name: str = "Brian"):
        """
        sizes: [2,10,10,2] for creating a NN with In/Output of size 2, and hidden layers of size 10.
        Weights and biases are set random if not specified further.
        Select one of the following non-linear functions: 'sigmoid' (default), 'ReLU', 'tanh'
        """
        rng = default_rng()
        self.name = name
        self.sizes = sizes.copy()
        self.input_size = self.sizes[0]
        self.output_size = self.sizes[-1]
        self.L = len(self.sizes) - 1  # Number of layers (excluding input)
        self.nlf = non_linear_funktion
        self.z = []
        self.activations = []
        
        if weights is None:
            self.weights = []
            for l in range(self.L):
                if l == 0:
                    # First layer: connect input to first hidden layer
                    w = rng.normal(0, 1, (self.sizes[l+1], self.sizes[l])) * 0.1
                else:
                    # Subsequent layers
                    w = rng.normal(0, 1, (self.sizes[l+1], self.sizes[l])) * 0.1
                self.weights.append(w)
        else:
            self.weights = weights
            
        if biases is None:
            self.biases = []
            for l in range(self.L):
                b = rng.normal(0, 1, (self.sizes[l+1],)) * 0.1
                self.biases.append(b)
        else:
            self.biases = biases

    def forward_prop(self, x) -> np.ndarray:
        """Forward propagation through the network"""
        x = np.array(x, dtype=float)
        self.z = []
        self.activations = [x]  # Store input as first activation
        
        a = x
        for l in range(self.L):
            z = np.dot(self.weights[l], a) + self.biases[l]
            self.z.append(z)
            
            if l == self.L - 1:  # Output layer
                # Use softmax for multi-class classification
                a = softmax(z)
            else:  # Hidden layers
                a = self.sigma(z)
            
            self.activations.append(a)
        
        return a

    def back_prop(self, output, y, learning_rate):
        """Backpropagation algorithm"""
        y = np.array(y, dtype=float)
        
        # Calculate output layer error
        if len(y) > 1:  # Multi-class
            delta = output - y  # For softmax + cross-entropy
        else:  # Binary classification
            delta = (output - y) * self.sigma_prime(self.z[-1])
        
        # Backpropagate the error
        for l in range(self.L - 1, -1, -1):
            # Update weights and biases
            self.weights[l] -= learning_rate * np.outer(delta, self.activations[l])
            self.biases[l] -= learning_rate * delta
            
            # Calculate error for previous layer (if not input layer)
            if l > 0:
                delta = np.dot(self.weights[l].T, delta) * self.sigma_prime(self.z[l-1])

    def train_on(self, data: list, iter: int = 100, learning_rate: float = 0.01):
        """
        Train the network on data.
        data: list of tuples [(x0, y0), (x1, y1), ...] 
        where y is the expected output to input x.
        """
        for i in range(iter):
            print(f"Epoch {i+1} of {iter}")
            total_loss = 0
            
            for x, y in data:
                x = np.array(x, dtype=float)
                y = np.array(y, dtype=float)
                
                # Forward pass
                output = self.forward_prop(x)
                
                # Calculate loss for monitoring
                if len(y) > 1:  # Multi-class
                    loss = -np.sum(y * np.log(output + 1e-15))  # Cross-entropy
                else:  # Binary
                    loss = 0.5 * np.sum((output - y) ** 2)  # MSE
                total_loss += loss
                
                # Backward pass
                self.back_prop(output, y, learning_rate)
            
            avg_loss = total_loss / len(data)
            if i % 10 == 0 or i == iter - 1:
                print(f"  Average loss: {avg_loss:.6f}")

    def sigma(self, x) -> np.ndarray:
        """Activation function"""
        match self.nlf:
            case "sigmoid":
                return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
            case "ReLU":
                return np.maximum(0, x)
            case "tanh":
                return np.tanh(x)

    def sigma_prime(self, x) -> np.ndarray:
        """Derivative of activation function"""
        match self.nlf:
            case "sigmoid":
                s = self.sigma(x)
                return s * (1 - s)
            case "ReLU":
                return np.where(x > 0, 1, 0).astype(float)
            case "tanh":
                t = np.tanh(x)
                return 1 - t * t

    def predict(self, x):
        """Make a prediction and return the class index"""
        output = self.forward_prop(x)
        return np.argmax(output)

    def print_parameters(self):
        """Print network weights and biases"""
        print("Weights:\n")
        for l in range(self.L):
            print(f"Layer {l}:\n{self.weights[l]}")
        print("\nBiases:\n")
        for l in range(self.L):
            print(f"Layer {l}:\n{self.biases[l]}")

# Example usage
if __name__ == "__main__":
    # Create network
    nn = NeuronalNetwork([9, 20, 20, 20, 20, 9], non_linear_funktion="tanh")
    
    # print("output before training:", nn.forward_prop([1, 2, 0, 1, 0, 0, 0, 0, 0]))
    # print("Predicted class before training:", nn.predict([1, 2, 0, 1, 0, 0, 0, 0, 0]))
    
    # Load or generate data
    try:
        with open('data.json', 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} training samples from data.json")
    except FileNotFoundError:
        print("data.json not found, generating new dataset...")
        data_new = dg.generate_dataset(1000)
        with open('data.json', 'w') as f:
            json.dump(data_new, f)
        data = data_new
        print(f"Generated and saved {len(data)} training samples")
    
    # Train the network
    nn.train_on(data, 200, 0.01)
    
    # Test after training
    test_input = [1, 2, 0,
                  1, 2, 0,
                  0, 0, 0]
    output = nn.forward_prop(test_input)
    pred_field = nn.predict(test_input)
    
    print(f"\noutput after training: {output}")
    print(f"Predicted field after training: {pred_field}")
    print(f"Confidence: {np.max(output):.4f}")