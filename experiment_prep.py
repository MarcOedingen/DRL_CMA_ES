import os
import re

def prepare_experiment_folders():
    print("------------------ Preparing experiment folders ------------------")
    changes_made = False

    results_policies_dir = "Results/Policies"
    if not os.path.exists(results_policies_dir):
        os.makedirs(results_policies_dir)
        print(f"Created folder: {results_policies_dir}")
        changes_made = True

    for dir in os.listdir("Environments"):
        policies_path = f"Environments/{dir}/Policies"
        samples_path = f"Environments/{dir}/Samples"

        if not os.path.exists(policies_path):
            os.makedirs(policies_path)
            print(f"Created folder: {policies_path}")
            changes_made = True
        else:
            for file in os.listdir(policies_path):
                next_index = get_next_index(file[:-4], results_policies_dir)
                new_file_name = f"{file[:-4]}_{next_index}.pkl"
                new_file_path = os.path.join(results_policies_dir, new_file_name)
                os.rename(f"Environments/{dir}/Policies/{file}", new_file_path)
                print(f"Moved file to: {new_file_path}")
                changes_made = True

        if not os.path.exists(samples_path):
            os.makedirs(samples_path)
            print(f"Created folder: {samples_path}")
            changes_made = True
        else:
            for file in os.listdir(samples_path):
                os.remove(f"Environments/{dir}/Samples/{file}")
                print(f"Removed file: {samples_path}/{file}")
                changes_made = True

    if not changes_made:
        print("Nothing changed")
    else:
        print("All folders exist and necessary changes made")

def get_next_index(file_name, directory):
    pattern = re.compile(re.escape(file_name) + r"_(\d+)")
    max_index = -1
    for existing_file in os.listdir(directory):
        match = pattern.match(existing_file)
        if match:
            index = int(match.group(1))
            max_index = max(max_index, index)
    return max_index + 1

prepare_experiment_folders()
