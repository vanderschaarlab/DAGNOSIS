from pathlib import Path

import dill
import numpy as np

PROJECT_ROOT = Path().resolve().parent.parent
import sys

sys.path.append(str(PROJECT_ROOT))


with open("artifacts_adult/artifacts_downstream", "rb") as f:
    dic_results = dill.load(f)



list_accuracy = dic_results["accuracy"] 
detection_all=  dic_results["detection_all"]
detection_on_sex=  dic_results["detection_on_sex"]
detection =   dic_results["detection"]


k = 5
detected_women_dagnosis = np.array(detection_all[1])[k]
detected_women_pca = np.array(detection_all[2])[k]

detected_all_dagnosis = np.array(detection[1])[k]
detected_all_pca = np.array(detection[2])[k]

detected_men_dagnosis = detected_all_dagnosis-detected_women_dagnosis
detected_men_pca = detected_all_pca - detected_women_pca

n_test = 15264 + 1000*k
print("DAGNOSIS: Proportion of Dtest flagged and consists of women", detected_women_dagnosis / n_test)
print("DATASUITE: Proportion of Dtest flagged and consists of women", detected_women_pca / n_test)
print("DAGNOSIS: Proportion of Dtest flagged and consists of men", detected_men_dagnosis/ n_test)
print("DATASUITE: Proportion of Dtest flagged and consists of men", detected_men_pca/n_test)



