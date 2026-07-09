# [Intermediate ML](https://www.kaggle.com/code/alexisbcook/introduction)

- preqrequisite
  - know overfitting and underfitting
  - know model validation
  - have a basic idea of random forest and decision trees

# exercise 1

- load full data from train
- load full test data from test
- get a target (column from train) set to Y
- set features (list of columns)
- take columns from train using features - set to X
- take columns from test using features - set to X_test
- split using train_test_split(X, Y, random_state=1) -> results in X_train, X_valid, Y_train, Y_valid
- 5 RF models created with different n_estimators, different criterion, different min_samples_split, max_depth
- run fit, mean_abslute_error on each model

# Missing Values

