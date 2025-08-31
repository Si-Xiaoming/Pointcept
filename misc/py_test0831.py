import numpy as np

def test_01():
    scale = (0.4, 1.0)
    max_size = 100
    size = int(np.random.uniform(*scale) * max_size)
    print(size)


if __name__ == '__main__':
    test_01()