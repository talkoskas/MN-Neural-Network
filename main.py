import pandas as pd
import numpy as np
from NN import NeuralNetwork
from sklearn.model_selection import train_test_split

def preprocess(filepath):
    df = pd.read_csv(filepath)
    y = df.loc[:, 'y']
    df.drop(columns="y", inplace=True)
    return df, y


def main():
    X_train, y_train = preprocess('MNIST-train.csv')
    X_test, y_test = preprocess('MNIST-test.csv')

    # Normalize your input data
    print("X_train std\n", X_train.std())
    train_std = X_train.std()
    test_std = X_test.std()
    train_std = np.where(train_std == 0, 1, train_std)
    test_std = np.where(test_std == 0, 1, test_std)
    X_train_normalized = (X_train - X_train.mean()) / train_std
    X_test_normalized = (X_test - X_test.mean()) / test_std
    print("X_train after normalize - \n", X_train_normalized)
    print(f"X_test after normalize - \n{X_test_normalized}")
    X_train, X_val, y_train, y_val = train_test_split(X_train_normalized, y_train, test_size=0.2, random_state=42)

    # Create and train the network
    input_size = X_train.shape[1]
    hidden_sizes = [100, 50]  # Two hidden layers with 100 and 50 neurons
    output_size = 10  # Assuming 10 classes
    # Train with early stopping and validation data
    nn = NeuralNetwork(input_size, hidden_sizes, output_size, learning_rate=0.001, reg_lambda=0.01)
    # Train with early stopping and validation data
    nn.train(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_val, y_val), early_stopping_patience=5)

    # Evaluate the network
    test_accuracy = nn.score(X_test_normalized, y_test)
    print(f"Final Test Accuracy: {test_accuracy:.4f}")


if __name__ == '__main__':
    main()
