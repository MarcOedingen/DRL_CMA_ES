import os
import numpy as np

def load_results(path):
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        return data["diff"].tolist()
    else:
        return []


def run(policy, dimension, instance):
    print(f"---------------Evaluating results for {policy} on {dimension}D-{instance}I---------------")
    path = f"Results/{policy}_{dimension}D_{instance}I.npz"
    results = load_results(path)
    print(f"Mean difference: {np.mean(results)} ± {np.std(results)} for {len(results)} test functions")

