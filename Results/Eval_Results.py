import os
import numpy as np

def load_results(path):
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        return data["diff"].tolist()
    else:
        return []


def run(policy, dimension, instance, split, p_class):
    p_class = p_class if split == "classes" else -1
    print(f"---------------Evaluating results for {policy} on {dimension}D_{instance}I_{p_class}C---------------")
    path = f"Results/{policy}_{dimension}D_{instance}I_{p_class}C.npz"
    results = load_results(path)
    print(f"Mean difference: {np.mean(results)} ± {np.std(results)} for {len(results)} test functions")

