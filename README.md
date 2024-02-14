# DAGnosis: Localized Identification of Data Inconsistencies using Structures

This repository accompanies the AISTATS'24 paper: "DAGnosis: Localized Identification of Data Inconsistencies using Structures".

### Usage
We suggest creating a new environment before using the code, e.g. with:
```shell
conda create --name dagnosis python=3.7.15
```
We can then install the requirements with:
```shell
python -m pip install -r requirements.txt
```

#### Synthetic
We illustrate how to use DAGnosis in a synthetic setup, via the files in the folder ```experiments/synthetic```.
The bash scripts ```run_linear.sh``` and ```run_mlp.sh``` run the full pipeline: generate the data, train the conformal estimators, and test the conformal estimators, for linear and MLP SEMs respectively.

To compute the inconsistency detection metrics (F1, Precision, Recall), go to the folder ```experiments/synthetic``` and run:
```shell
python compute_metrics.py PATH_SAVE_METRIC=path_metrics
```
where ```path_metrics``` denotes the folder where the metrics are saved.

Similarly, you can reproduce the sensitivity experiment by going to the folder ```experiments/synthetic/sensitivity``` and using the script ```run.sh```, followed by
```shell
python compute_metrics.py PATH_SAVE_METRIC=path_metrics
```

#### UCI Adult Income
To run the experiments on the UCI Adult Income dataset, go to the folder ```experiments/adult```.
In order to train and test the conformal estimators, run
```shell
python train_test_adult.py
```
The artifacts will be saved in the folder ```artifacts_adult```.
Then, the results can be obtained by executing:
```shell
python proportion_flagging.py
```
which will print the list of downstream accuracies and proportions of samples flagged (Figure 3 a) and b)).


### Citing
If you use this software, please cite the original paper:
```shell
TODO
```
