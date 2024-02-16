# stdlib
import sys
from pathlib import Path

# third party
import dill
import numpy as np

PROJECT_ROOT = Path().resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))


with open("artifacts_adult/artifacts_final", "rb") as f:
    dic_results = dill.load(f)


list_accuracy = dic_results["accuracy"]
detection_all = dic_results["detection_all"]
detection_on_sex = dic_results["detection_on_sex"]
detection = dic_results["detection"]


k = 5

accuracies_dagnosis = list_accuracy[1]
accuracies_pca = list_accuracy[2]
accuracies_noflagging = list_accuracy[3]

print("List of accuracies (varying k)")

print("DAGNOSIS", accuracies_dagnosis)
print("DATASUITE", accuracies_pca)
print("NO FLAGGING", accuracies_noflagging)

detected_women_dagnosis = np.array(detection_all[1])[k]
detected_women_pca = np.array(detection_all[2])[k]

detected_all_dagnosis = np.array(detection[1])[k]
detected_all_pca = np.array(detection[2])[k]

detected_men_dagnosis = detected_all_dagnosis - detected_women_dagnosis
detected_men_pca = detected_all_pca - detected_women_pca

print("Proportions")

n_test = 15264 + 1000 * k
print(
    "DAGNOSIS: Proportion of Dtest flagged and consists of women",
    detected_women_dagnosis / n_test,
)
print(
    "DATASUITE: Proportion of Dtest flagged and consists of women",
    detected_women_pca / n_test,
)
print(
    "DAGNOSIS: Proportion of Dtest flagged and consists of men",
    detected_men_dagnosis / n_test,
)
print(
    "DATASUITE: Proportion of Dtest flagged and consists of men",
    detected_men_pca / n_test,
)
