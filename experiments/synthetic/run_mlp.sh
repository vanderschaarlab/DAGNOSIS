python generate_data.py sem_type=mlp PATH_SAVE_DATA=artifacts/mlpSEM_data/ PATH_SAVE_CP=artifacts/mlpSEM_cp/ PATH_SAVE_CONFDICT=artifacts/mlpSEM_confdict/ PATH_SAVE_METRIC=artifacts/mlpSEM_metric/
python train_cp.py PATH_SAVE_DATA=artifacts/mlpSEM_data/ PATH_SAVE_CP=artifacts/mlpSEM_cp/ PATH_SAVE_CONFDICT=artifacts/mlpSEM_confdict/ PATH_SAVE_METRIC=artifacts/mlpSEM_metric/
python test_cp.py PATH_SAVE_DATA=artifacts/mlpSEM_data/ PATH_SAVE_CP=artifacts/mlpSEM_cp/ PATH_SAVE_CONFDICT=artifacts/mlpSEM_confdict/ PATH_SAVE_METRIC=artifacts/mlpSEM_metric/
