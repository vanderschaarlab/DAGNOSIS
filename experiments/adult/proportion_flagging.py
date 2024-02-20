# stdlib
import sys
from pathlib import Path

# third party
import dill
import numpy as np
import pandas as pd

PROJECT_ROOT = Path().resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

acc_table = []
proportions_table = []


for seed in range(5):
    with open(f"artifacts_adult/artifacts_final_seed_{seed}", "rb") as f:
        dic_results = dill.load(f)

    list_accuracy = dic_results["accuracy"]
    detection_all = dic_results["detection_all"]
    detection_on_sex = dic_results["detection_on_sex"]
    detection = dic_results["detection"]

    accuracies_dagnosis = list_accuracy[1]
    accuracies_pca = list_accuracy[2]
    accuracies_noflagging = list_accuracy[3]

    for j in range(len(accuracies_dagnosis)):
        acc_table.append(
            {
                "seed": seed,
                "method": "dagnosis",
                "k": j,
                "accuracy": accuracies_dagnosis[j],
            }
        )
        acc_table.append(
            {"seed": seed, "method": "pca", "k": j, "accuracy": accuracies_pca[j]}
        )
        acc_table.append(
            {
                "seed": seed,
                "method": "noflagging",
                "k": j,
                "accuracy": accuracies_noflagging[j],
            }
        )

    k = 5
    n_test = 15264 + 1000 * k

    detected_women_dagnosis = np.array(detection_all[1])[k]
    detected_women_pca = np.array(detection_all[2])[k]

    detected_all_dagnosis = np.array(detection[1])[k]
    detected_all_pca = np.array(detection[2])[k]

    detected_men_dagnosis = detected_all_dagnosis - detected_women_dagnosis
    detected_men_pca = detected_all_pca - detected_women_pca

    proportions_table.append(
        {
            "seed": seed,
            "method": "dagnosis",
            "k": k,
            "f": "women",
            "proportion": detected_women_dagnosis / n_test,
        }
    )
    proportions_table.append(
        {
            "seed": seed,
            "method": "dagnosis",
            "k": k,
            "f": "men",
            "proportion": detected_men_dagnosis / n_test,
        }
    )

    proportions_table.append(
        {
            "seed": seed,
            "method": "pca",
            "k": k,
            "f": "women",
            "proportion": detected_women_pca / n_test,
        }
    )
    proportions_table.append(
        {
            "seed": seed,
            "method": "pca",
            "k": k,
            "f": "men",
            "proportion": detected_men_pca / n_test,
        }
    )


# convert to dataframe
df_accuracy = pd.DataFrame(acc_table)

# for each method, print the mean accuracy and the standard deviation for each k
for method in ["dagnosis", "pca", "noflagging"]:
    print(f"Method: {method}")
    for j in range(len(accuracies_dagnosis)):
        df = df_accuracy[(df_accuracy["method"] == method) & (df_accuracy["k"] == j)][
            "accuracy"
        ]
        mean_accuracy = df.mean()
        ci_accuracy = 1.96 * df.std() / np.sqrt(len(df))
        #print(f"K = {j}, mean accuracy = {mean_accuracy}, 1.96*SE = {ci_accuracy}")
        
        #For the tikz plot
        print(f"({j},{mean_accuracy}) +=(0,{ci_accuracy}) -=(0,{ci_accuracy})")

print("Proportions")
df_proportions = pd.DataFrame(proportions_table)

for method in ["dagnosis", "pca"]:
    for f in ["women", "men"]:
        df = df_proportions[(df_proportions["method"] == method) & (df_proportions["f"] == f)]
        print(
            f"Method: {method}, f: {f}, mean proportion = {df['proportion'].mean()}, 1.96*SE = {1.96*df['proportion'].std()/np.sqrt(len(df))}"
        )
