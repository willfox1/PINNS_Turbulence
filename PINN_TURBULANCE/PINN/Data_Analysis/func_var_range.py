import numpy as np
import os
import matplotlib.pyplot as plt

def load_data(filename):
    """Load data from file and return u, v, p."""
    data = np.loadtxt(filename, skiprows=20)
    u = data[:, 2]
    v = data[:, 3]
    p = data[:, 5]
    return u, v, p

def calculate_stats(data):
    """Calculate mean and standard deviation of data."""
    return np.mean(data), np.std(data)

def plot_histogram(ax, data, color, label):
    """Plot histogram on given axis."""
    ax.hist(data, bins=30, density=True, color=color, alpha=0.5, label=label)

def plot_normal_curve(ax, data, color, label):
    """Plot normal curve on given axis."""
    mean, std = calculate_stats(data)
    x_values = np.linspace(min(data), max(data), 100)
    ax.plot(x_values, 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x_values - mean)**2 / (2 * std**2) ), color=color, label=label)

def set_plot_properties(ax, title, xlabel):
    """Set title, xlabel and grid for given axis."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density (%)')
    ax.grid(True)
    ax.legend()

def main():
    filename = 'DNS_29_Projects/alph05-4071-3036.dat'
    u, v, p = load_data(filename)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    plot_histogram(axs[0], u, 'blue', 'Histogram')
    plot_normal_curve(axs[0], u, 'blue', 'Normal Curve')
    set_plot_properties(axs[0], 'u', 'Streamwise Velocity (m/s)')

    plot_histogram(axs[1], v, 'orange', 'Histogram')
    plot_normal_curve(axs[1], v, 'orange', 'Normal Curve')
    set_plot_properties(axs[1], 'v', 'Transverse Velocity (m/s)')

    plot_histogram(axs[2], p, 'green', 'Histogram')
    plot_normal_curve(axs[2], p, 'green', 'Normal Curve')
    set_plot_properties(axs[2], 'p', 'Pressure (Pa)')

    plt.tight_layout()
    plt.savefig('Project/figs/data_distribution.svg', format='svg')
    plt.show()

if __name__ == "__main__":
    main()
