import numpy as np
import time


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01, reg_lambda=0.01):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.Z = {}
        self.A = {}
        # Initialize weights and biases
        self.parameters = {}
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, len(layer_sizes)):
            self.parameters[f'W{i}'] = np.random.randn(layer_sizes[i - 1], layer_sizes[i]) * np.sqrt(
                2. / layer_sizes[i - 1])
            self.parameters[f'b{i}'] = np.zeros((1, layer_sizes[i]))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return Z > 0

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def stable_softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def softmax_derivative(self, x):
        s = self.softmax(x)
        return s * (1 - s)

    def forward(self, X):
        if hasattr(X, 'values'):
            X = X.values

        self.Z = {}
        self.A = {0: X}

        # Zs on the 1st iteration is failing.
        for i in range(1, len(self.hidden_sizes) + 2):
            """print(f"A[{i-1}] = \n{self.A[i-1]}")
            print(f"params[f'W{i}'] = \n {self.parameters[f'W{i}']}")
            print(f"params[f'b{i}'] = \n {self.parameters[f'b{i}']}")"""
            self.Z[i] = self.A[i - 1] @ self.parameters[f'W{i}'] + self.parameters[f'b{i}']
            # print(f"Z[{i}]=\n{self.Z[i]}")
            self.A[i] = self.relu(self.Z[i]) if i < len(self.hidden_sizes) + 1 else self.stable_softmax(self.Z[i])
            # print(f"A[{i}]=\n{self.A[i]}")

        return self.A[len(self.hidden_sizes) + 1]

    def backward(self, X, Y):
        m = X.shape[0]
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(Y, 'values'):
            Y = Y.values

        dZ = self.A[len(self.hidden_sizes) + 1] - Y

        for i in range(len(self.hidden_sizes) + 1, 0, -1):
            if hasattr(dZ, 'values'):
                dZ = dZ.values
            if hasattr(self.A[i - 1], 'values'):
                self.A[i - 1] = self.A[i - 1].values

            dW = (1 / m) * (self.A[i - 1].T @ dZ) + (self.reg_lambda / m) * self.parameters[f'W{i}']
            db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)

            # Gradient clipping
            dW = np.clip(dW, -1, 1)
            db = np.clip(db, -1, 1)

            if i > 1:
                dA = dZ @ self.parameters[f'W{i}'].T
                dZ = dA * self.relu_derivative(self.Z[i - 1])

            self.parameters[f'W{i}'] -= self.learning_rate * dW
            self.parameters[f'b{i}'] -= self.learning_rate * db
    def train(self, X, y, epochs, batch_size=32, validation_data=None, early_stopping_patience=5):
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        y_one_hot = np.eye(self.output_size)[y]
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()

        for epoch in range(epochs):
            # Shuffle the data
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y_one_hot[indices]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                self.forward(X_batch)
                self.backward(X_batch, y_batch)

            # Compute and print metrics every 10 epochs
            if epoch % 10 == 0:
                train_loss = self.compute_loss(X, y_one_hot)
                train_accuracy = self.score(X, y)

                print(f"Epoch {epoch}/{epochs}, Time: {time.time() - start_time:.2f}s")
                print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

                if validation_data:
                    X_val, y_val = validation_data
                    if hasattr(X_val, 'values'):
                        X_val = X_val.values
                    if hasattr(y_val, 'values'):
                        y_val = y_val.values
                    val_loss = self.compute_loss(X_val, np.eye(self.output_size)[y_val])
                    val_accuracy = self.score(X_val, y_val)
                    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= early_stopping_patience:
                        print("Early stopping")
                        return
                    if train_accuracy > 0.985 and val_accuracy > 0.96:
                        print("Early stopping")
                        return

    def compute_loss(self, X, y):
        y_pred = self.forward(X)
        print("y_pred in compute loss:\n", y_pred)
        print("y in compute loss:\n", y)
        return -np.mean(y * np.log(y_pred + 1e-8))

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)