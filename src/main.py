import numpy as np

from src.layers.layer import Linear, ReLU, NeuralNetwork, Sigmoid, MSE, Tanh
from data_loading.dataset import load_sin_set
import matplotlib.pyplot as plt


def get_network():
    layers = [
        Linear(1, 10),
        ReLU(),
        Linear(10, 1),
        # Sigmoid()
        Tanh()
    ]
    model = NeuralNetwork(layers)
    return model


def evaluate(model, x, y):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    out = model.forward(x)
    plt.plot(x, y, label='Ground Truth')
    plt.plot(x, out, label='Prediction')
    plt.legend()
    plt.show()


def main():
    model = get_network()
    criterion = MSE()
    epochs = 100
    batch_size = 10

    x_set, y_set = load_sin_set()
    plt.scatter(x_set, y_set, s=0.1)
    plt.show()
    # shuffle
    idx = np.arange(len(x_set))
    np.random.shuffle(idx)
    x_loader = x_set[idx]
    y_loader = y_set[idx]
    # make into [n, B, 1]
    x_loader = x_loader.reshape(-1, batch_size, 1)
    y_loader = y_loader.reshape(-1, batch_size, 1)

    for e in range(epochs):
        total = 0
        total_loss = 0
        for x, y in zip(x_loader, y_loader):
            total += len(y)
            out = model.forward(x)
            loss = criterion.forward(out, y)
            total_loss += loss
            model.backward(criterion.backward(loss))
            model.update(0.01)

        print(f'Epoch {e + 1}/{epochs} - Loss: {total_loss / total}')

    evaluate(model, x_set, y_set)


if __name__ == '__main__':
    main()
