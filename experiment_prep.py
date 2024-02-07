import os


def prepare_experiment_folders():
    print("------------------ Preparing experiment folders ------------------")
    for dir in os.listdir("Environments"):
        if not os.path.exists(f"Environments/{dir}/Policies"):
            os.makedirs(f"Environments/{dir}/Policies")
            print(f"Created folder: Environments/{dir}/Policies")
        else:
            for file in os.listdir(f"Environments/{dir}/Policies"):
                os.remove(f"Environments/{dir}/Policies/{file}")
                print(f"Removed file: Environments/{dir}/Policies/{file}")
        if not os.path.exists(f"Environments/{dir}/Samples"):
            os.makedirs(f"Environments/{dir}/Samples")
            print(f"Created folder: Environments/{dir}/Samples")
        else:
            for file in os.listdir(f"Environments/{dir}/Samples"):
                os.remove(f"Environments/{dir}/Samples/{file}")
                print(f"Removed file: Environments/{dir}/Samples/{file}")


prepare_experiment_folders()
