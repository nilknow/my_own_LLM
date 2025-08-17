import numpy as np
import matplotlib.pyplot as plt


def get_sinusoidal_position_encoding(max_len, d_model):
    position = np.arange(max_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe


if __name__ == '__main__':
    max_len = 50
    d_model = 64
    pe = get_sinusoidal_position_encoding(max_len, d_model)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.imshow(pe, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Sinusoidal Position Encoding")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")
    plt.show()
