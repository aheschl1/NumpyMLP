import numpy as np

from src.layers.layer import Linear, ReLU, NeuralNetwork, Sigmoid, MSE


def main():
    x = np.random.randn(1, 5)
    layers = [
        Linear(5, 3),
        ReLU(),
        Linear(3, 2),
        Sigmoid()
    ]
    model = NeuralNetwork(layers)
    criterion = MSE()
    target = np.array([[1., 1.]])
    for e in range(10000):
        out = model.forward(x)
        print(out)
        loss = criterion(out, target)

        model.backward(criterion.backward(loss))

        model.update(1)


if __name__ == '__main__':
    main()
