from torchvision import datasets


def load_sin_set():
    import numpy as np
    x = np.linspace(0, 2*np.pi, 1000).reshape(-1, 1)
    y = np.sin(x)
    return x, y
