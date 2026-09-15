import numpy as np


class MLP:
    def __init__(self, n_hidden, eta=0.001, seed=42):
        rng = np.random.default_rng(seed)

        # Input -> hidden
        self.W1 = rng.normal(0, 0.1, size=(1, n_hidden))
        self.b1 = np.zeros(n_hidden)

        # Hidden -> output
        self.W2 = rng.normal(0, 0.1, size=n_hidden)
        self.b2 = 0.0

        self.eta = eta


    def forward(self, X):
        X = np.asarray(X).reshape(-1, 1)

        # Hidden layer
        z1 = X @ self.W1 + self.b1
        h = np.tanh(z1)

        # Linear output layer
        y = h @ self.W2 + self.b2

        return y, h


    def train(self, X, targets, epochs=3000):
        X = np.asarray(X).reshape(-1, 1)
        targets = np.asarray(targets)

        n = len(X)

        for epoch in range(epochs):

            # Forward
            z1 = X @ self.W1 + self.b1
            h = np.tanh(z1)
            predictions = h @ self.W2 + self.b2

            # Error
            error = predictions - targets

            # Output layer gradients
            dW2 = (h.T @ error) / n
            db2 = np.mean(error)

            # Hidden layer gradients
            dh = np.outer(error, self.W2)
            dz1 = dh * (1 - h**2)

            dW1 = (X.T @ dz1) / n
            db1 = np.mean(dz1, axis=0)

            # Gradient descent
            self.W2 -= self.eta * dW2
            self.b2 -= self.eta * db2

            self.W1 -= self.eta * dW1
            self.b1 -= self.eta * db1


    def predict(self, X):
        predictions, _ = self.forward(X)
        return predictions