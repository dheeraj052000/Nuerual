#!/usr/bin/env python3
"""
CS5720 Assignment 1: Neural Network Fundamentals
Starter Code - Build a Neural Network from Scratch

Instructions:
- Use only NumPy for computations
- Follow the docstring specifications carefully
- Run test_solution.py to verify your implementation
"""
"""
Student Name:Dheeraj Kandagatla
Student ID: 700756540 
"""

import numpy as np
import struct
import gzip
from typing import List, Tuple, Dict
import pickle
from sklearn.model_selection import train_test_split

# ============================================================================
# Data Loading Utilities
# ============================================================================

def load_mnist(path='data/'):
    """
    Load MNIST dataset from files or download if not present.
    
    Returns:
        X_train, y_train, X_test, y_test as numpy arrays
    """
    import os
    import urllib.request
    
    # Create data directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    
    # MNIST file information
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    # Download files if not present
    base_url = 'https://github.com/fgnt/mnist/'
    for file in files.values():
        filepath = os.path.join(path, file)
        if not os.path.exists(filepath):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(base_url + file, filepath)
    
    # Load data
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num, rows * cols)
            return images / 255.0  # Normalize to [0, 1]
    
    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
    
    X_train = load_images(os.path.join(path, files['train_images']))
    y_train = load_labels(os.path.join(path, files['train_labels']))
    X_test = load_images(os.path.join(path, files['test_images']))
    y_test = load_labels(os.path.join(path, files['test_labels']))
    
    return X_train, y_train, X_test, y_test


def one_hot_encode(y, num_classes=10):
    """Convert integer labels to one-hot encoding."""
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


# ============================================================================
# Layer Implementations
# ============================================================================

class Layer:
    """Base class for all layers."""
    def forward(self, X):
        self.X=X
    
    def backward(self, dL_dY):
        self.dL_dY=dL_dY
    
    def get_params(self):
        return {}
    
    def get_grads(self):
        return {}
    
    def set_params(self, params):
        pass


class Dense(Layer):
    """
    Fully connected (dense) layer.
    
    Parameters:
        input_dim: Number of input features
        output_dim: Number of output features
        weight_init: Weight initialization method ('xavier', 'he', 'normal')
    """
    def __init__(self, input_dim, output_dim, weight_init='xavier'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        if weight_init == 'xavier':
            limit = np.sqrt(2 / (input_dim + output_dim))
            self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        elif weight_init == 'he':
            limit = np.sqrt(2 / input_dim)
            self.W = np.random.randn(input_dim, output_dim) * limit
        else:  
            self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros(output_dim)
        
        # Storage for backward pass
        self.X = None
        self.dW = None
        self.db = None
    
    def forward(self, X):
        """
        Forward pass: Y = XW + b
        
        Args:
            X: Input data, shape (batch_size, input_dim)
            
        Returns:
            Y: Output data, shape (batch_size, output_dim)
        """
        
        # Store X for backward pass
        self.X = X
        return X @ self.W + self.b
    
    def backward(self, dL_dY):
        """
        Backward pass: compute gradients.
        
        Args:
            dL_dY: Gradient of loss w.r.t. output, shape (batch_size, output_dim)
            
        Returns:
            dL_dX: Gradient of loss w.r.t. input, shape (batch_size, input_dim)
        """
        
        self.dW = self.X.T @ dL_dY
        self.db = np.sum(dL_dY, axis=0)
        dL_dX = dL_dY @ self.W.T
        return dL_dX
    
    def get_params(self):
        return {'W': self.W, 'b': self.b}
    
    def get_grads(self):
        return {'W': self.dW, 'b': self.db}
    
    def set_params(self, params):
        self.W = params['W']
        self.b = params['b']


# ============================================================================
# Activation Functions
# ============================================================================

class Activation(Layer):
    """Base class for activation functions."""
    def __init__(self):
        self.cache = None


class ReLU(Activation):
    """Rectified Linear Unit activation function."""
    
    def forward(self, X):
        """
        Forward pass: f(x) = max(0, x)
        
        Args:
            X: Input data
            
        Returns:
            Output after applying ReLU
        """
        
        # Store input for backward pass
        self.X= X
        return np.maximum(0, X)
    
    def backward(self, dL_dY):
        """
        Backward pass: f'(x) = 1 if x > 0 else 0
        
        Args:
            dL_dY: Gradient of loss w.r.t. output
            
        Returns:
            dL_dX: Gradient of loss w.r.t. input
        """
        
        dX = dL_dY * (self.X > 0)
        return dX


class Sigmoid(Activation):
    """Sigmoid activation function."""
    def __init__(self):
        self.Y = None
    
    def forward(self, X):
        """
        Forward pass: f(x) = 1 / (1 + exp(-x))
        
        Args:
            X: Input data
            
        Returns:
            Output after applying sigmoid
        """
        
        # Store output for backward pass
        self.Y = 1 / (1 + np.exp(-X))
        return self.Y
    
    def backward(self, dL_dY):
        """
        Backward pass: f'(x) = f(x) * (1 - f(x))
        
        Args:
            dL_dY: Gradient of loss w.r.t. output
            
        Returns:
            dL_dX: Gradient of loss w.r.t. input
        """
        
        dX = dL_dY * self.Y * (1 - self.Y)
        return dX


class Softmax(Activation):
    """Softmax activation function."""
    def __init__(self):
        self.Y = None
    
    def forward(self, X):
        """
        Forward pass: f(x_i) = exp(x_i) / sum(exp(x))
        
        Args:
            X: Input data, shape (batch_size, num_classes)
            
        Returns:
            Output probabilities, shape (batch_size, num_classes)
        """
        
        # Hint: Subtract max for numerical stability
        X_stable = X - np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X_stable)
        self.Y = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.Y
    
    def backward(self, dL_dY):
        """
        Backward pass for softmax.
        
        Args:
            dL_dY: Gradient of loss w.r.t. output
            
        Returns:
            dL_dX: Gradient of loss w.r.t. input
        """
        
        # This is complex - consider the Jacobian matrix
        # For cross-entropy + softmax, gradient is simplified
        S = self.Y.T
        dL_dX = np.zeros_like(self.Y)
        for i in range(self.Y.shape[0]):
            jacobian = np.diag(self.Y[i]) - np.outer(self.Y[i], self.Y[i])
            dL_dX[i] = dL_dY[i] @ jacobian
            
        return dL_dX


# ============================================================================
# Loss Functions
# ============================================================================

class Loss:
    """Base class for loss functions."""
    def compute(self, y_pred, y_true):
        self.y_pred=y_pred
        self.y_true=y_true
    
    def gradient(self, y_pred, y_true):
        self.y_pred=y_pred
        self.y_true=y_true


class MSELoss(Loss):
    """Mean Squared Error loss."""
    
    def compute(self, y_pred, y_true):
        """
        Compute MSE loss: L = 0.5 * mean((y_pred - y_true)^2)
        
        Args:
            y_pred: Predictions, shape (batch_size, num_features)
            y_true: True values, shape (batch_size, num_features)
            
        Returns:
            Scalar loss value
        """
        mse=0.5 * np.mean((y_pred - y_true) ** 2)
        return mse
    
    def gradient(self, y_pred, y_true):
        """
        Compute gradient of MSE loss.
        
        Args:
            y_pred: Predictions
            y_true: True values
            
        Returns:
            Gradient w.r.t. predictions
        """
       
        # dL/dy_pred = (y_pred - y_true) / batch_size
        total_elements = y_pred.shape[0] * y_pred.shape[1]
        gradient=(y_pred - y_true) / total_elements
        return gradient


class CrossEntropyLoss(Loss):
    """Cross-entropy loss for classification."""
    
    def compute(self, y_pred, y_true):
        """
        Compute cross-entropy loss: L = -mean(sum(y_true * log(y_pred)))
        
        Args:
            y_pred: Predicted probabilities, shape (batch_size, num_classes)
            y_true: True labels (one-hot), shape (batch_size, num_classes)
            
        Returns:
            Scalar loss value
        """
    
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def gradient(self, y_pred, y_true):
        gradient=(y_pred - y_true) / y_pred.shape[0]
        return gradient


# ============================================================================
# Optimizers
# ============================================================================

class Optimizer:
    """Base class for optimizers."""
    def update(self, params, grads):
        self.params=params
        self.grads=grads


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        """
        Update parameters using vanilla SGD.
        
        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients
        """
        
        # params = params - learning_rate * grads
        for key in params.keys():
            params[key] = params[key] - self.learning_rate * grads[key]
        return params


class Momentum(Optimizer):
    """SGD with momentum optimizer."""
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate= learning_rate
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, params, grads):
        """
        Update parameters using SGD with momentum.

        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients
        """
        for key in params.keys():
            param_array = params[key]
            grad_array = grads[key]
            
            # The key for velocity must be globally unique to the parameter.
            # This is handled by the caller (NeuralNetwork).
            
            # Initialize velocity if not already.
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(param_array)
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate* grad_array
            params[key] = param_array + self.velocity[key]
        
        return params
# ============================================================================
# Neural Network Class (continued)
# ============================================================================

class NeuralNetwork:
    """
    Modular neural network implementation.
    
    Example usage:
        model = NeuralNetwork()
        model.add(Dense(784, 128))
        model.add(ReLU())
        model.add(Dense(128, 10))
        model.add(Softmax())
        model.compile(loss=CrossEntropyLoss(), optimizer=SGD(0.01))
        model.fit(X_train, y_train, epochs=10, batch_size=32)
    """
    
    def __init__(self):
        self.layers = []
        self.loss_fn = None
        self.optimizer = None
    
    def add(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)
    
    def compile(self, loss, optimizer):
        """Configure the model for training."""
        self.loss_fn = loss
        self.optimizer = optimizer
    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output of the network
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backward(self, dL_dY):
        """
        Backward propagation through all layers.
        
        Args:
            dL_dY: Gradient of loss w.r.t. network output
        """
        grad = dL_dY
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def update_params(self):
        """Update parameters of all trainable layers using the optimizer."""
        # Use optimizer to update parameters
        all_params = {}
        all_grads = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "get_params"):
                params = layer.get_params()
                grads = layer.get_grads()
                
                # Check if there are gradients to update
                if grads:
                    # Create unique keys for each parameter
                    for key, param in params.items():
                        unique_key = f'{i}_{key}'
                        all_params[unique_key] = param
                        all_grads[unique_key] = grads[key]
        
        # Now, call the optimizer's update method once with all parameters and gradients
        # The optimizer will handle its internal velocity state based on the unique keys.
        updated_params_dict = self.optimizer.update(all_params, all_grads)
        
        # Update the parameters back in each layer
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "get_params"):
                params = layer.get_params()
                for key in params.keys():
                    unique_key = f'{i}_{key}'
                    params[key] = updated_params_dict[unique_key]
                layer.set_params(params)
    
    def fit(self, X_train, y_train, epochs, batch_size, 
            X_val=None, y_val=None, verbose=True):
        """
        Train the neural network.
        
        Args:
            X_train: Training data
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch training
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            verbose: Print training progress
            
        Returns:
            Dictionary containing training history
        """
        history = {'train_loss': [], 'train_acc': [], 
                   'val_loss': [], 'val_acc': []}
        
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            
            # 1. Shuffle training data
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            # 2. Process mini-batches
            for i in range(0, n_samples, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                # 3. Forward pass
                y_pred = self.forward(X_batch)
                
                # 4. Compute loss
                loss = self.loss_fn.compute(y_pred, y_batch)
                
                # 5. Backward pass
                grad = self.loss_fn.gradient(y_pred, y_batch)
                self.backward(grad)
                
                # 6. Update parameters
                self.update_params()
            
            # 7. Track metrics
            train_pred = self.forward(X_train)
            train_loss = self.loss_fn.compute(train_pred, y_train)
            train_acc = np.mean(np.argmax(train_pred, axis=1) == np.argmax(y_train, axis=1))
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.loss_fn.compute(val_pred, y_val)
                val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
            else:
                val_loss, val_acc = None, None
            
            if verbose:
                val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
                val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "N/A"
                print(f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss_str}, Val Acc: {val_acc_str}")
            
        return history
    
    def predict(self, X):
        """
        Make predictions on input data.
        
        Args:
            X: Input data
            
        Returns:
            Predictions (class indices for classification)
        """
        
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance.
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            loss, accuracy
        """
        
        y_pred = self.forward(X)
        loss = self.loss_fn.compute(y_pred, y)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        return loss, accuracy
    def save_weights(self, filename):
        """Save model weights to file."""
        weights = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_params'):
                weights[f'layer_{i}'] = layer.get_params()
        np.savez(filename, **weights)
    
    def load_weights(self, filename):
        """Load model weights from file."""
        weights = np.load(filename)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'set_params') and f'layer_{i}' in weights:
                layer.set_params(weights[f'layer_{i}'])
# ============================================================================
# Gradient Checking Utility
# ============================================================================

def gradient_check(model, X, y, epsilon=1e-7):
    """
    Perform gradient checking for the model.
    """
    # Forward + backward pass to get analytical gradients
    y_pred = model.forward(X)
    grad = model.loss_fn.gradient(y_pred, y)
    model.backward(grad)

    results = {}
    batch_size = X.shape[0]  # Get batch size here
    
    for l in model.layers:
        if not hasattr(l, "get_params"):
            continue
        p, g = l.get_params(), l.get_grads()
        
        for k in p:
            it = np.nditer(p[k], flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                old = p[k][idx]
                
                # Calculate numerical gradient using finite differences
                p[k][idx] = old + epsilon
                plus = model.loss_fn.compute(model.forward(X), y)
                
                p[k][idx] = old - epsilon
                minus = model.loss_fn.compute(model.forward(X), y)
                
                p[k][idx] = old  # Restore original parameter value
                
                num = (plus - minus) / (2 * epsilon)
                
                # Normalize the analytical gradient for correct comparison
                analytical_grad = g[k][idx] 
                
                rel_err = abs(num - analytical_grad) / max(1e-8, abs(num) + abs(analytical_grad))
                
                if rel_err > 1e-5:
                    results[f"{l.__class__.__name__}_{k}_{idx}"] = rel_err 
                it.iternext()
                
    return results


# ============================================================================
# Main Training Script
# ============================================================================

if __name__ == "__main__":
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist()
    
    # Convert labels to one-hot encoding
    y_train_oh = one_hot_encode(y_train)
    y_test_oh = one_hot_encode(y_test)
    
    # ---- NEW: Split into training + validation sets ----
   
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train_oh, test_size=0.1, random_state=42
    )
    
    # Create model
    print("Building neural network...")
    model = NeuralNetwork()
    
    
    model.add(Dense(784, 128))
    model.add(ReLU())
    model.add(Dense(128, 64))
    model.add(ReLU())
    model.add(Dense(64, 10))
    model.add(Softmax())
    
    
    model.compile(loss=CrossEntropyLoss(), optimizer=Momentum(learning_rate=0.01, momentum=0.9))
    
    
    print("Training model...")
    history = model.fit(
        X_train_sub, y_train_sub,
        epochs=10, batch_size=32,
        X_val=X_val, y_val=y_val,
        verbose=True
    )
    
    
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test_oh)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    
    np.savez('model_weights.npz', *[layer.get_params() for layer in model.layers if hasattr(layer, "get_params")])
    
   
    with open('training_log.txt', 'w') as f:
        f.write("Training Log\n")
        f.write(f"History: {history}\n")
    predictions = model.predict(X_test[:100])
    np.savetxt('predictions_sample.txt', predictions, fmt='%d')
    
    print("Training complete!")


