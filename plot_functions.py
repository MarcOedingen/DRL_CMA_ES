import numpy as np
import matplotlib.pyplot as plt
from cocoex.function import BenchmarkFunction

# Setup for benchmark functions
func_ids = [
    16,
    21,
]  # IDs for Weierstrass and Gallagher's Gaussian 101-me Peaks, as examples
dimension = 3
instances = [1, 5, 10]
domain = [-5, 5]
n_points = int(1e3)
bbob_functions = [
    BenchmarkFunction("bbob", fid, dimension, i) for fid in func_ids for i in instances
]

x = np.linspace(domain[0], domain[1], n_points)
y = np.linspace(domain[0], domain[1], n_points)
X, Y = np.meshgrid(x, y)
Zs = []

# Function evaluation
for func in bbob_functions:
    Z = np.zeros((n_points, n_points))
    for j in range(n_points):
        for k in range(n_points):
            Z[j, k] = func(
                [X[j, k], Y[j, k], 0]
            )  # Assuming a 3D function with a fixed 3rd parameter
    Zs.append(Z)

# Function names for file naming
func_names = ["Weierstrass", "Gallaghers_Gaussian_101_me_Peaks"]

# Generate and save plots
for i, Z in enumerate(Zs):
    func_index = i // len(instances)  # Determine which function we're dealing with
    instance_index = i % len(instances)  # Determine which instance
    func_name = func_names[func_index]
    instance_id = instances[instance_index]

    plt.figure(figsize=(5, 5))
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, cmap="seismic")

    # Define the file name based on the function characteristics
    file_name = f"{func_name}_instance_{instance_id}_dimension_{dimension}.pdf"
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
    plt.close()  # Close the plt object to free up memory

    print(f"Saved: {file_name}")
