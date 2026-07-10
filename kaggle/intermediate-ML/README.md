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

- approaches to dealing with missing values
  1.  drop columns with missing values
  2.  imputation - fill missing values with some number like mean
  3.  imputation with flag indicating it was missing

# exercise 2
- df.drop(cols_with_missing) - to drop columns

- imputation done with a SimpleImputer() class
- my_imputer.fit_transform(x_train) - to fit and transform the training data

```py
# Fill in the line below: preprocess test data
cols_with_missing = [col for col in X_test.columns
                     if X_test[col].isnull().any()] # Your code here

# Fill in the lines below: drop columns in training and test data
final_X_train = X_train.drop(cols_with_missing, axis=1)
final_X_test = X_test.drop(cols_with_missing, axis=1)
# Fill in the line below: get test predictions
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(final_X_train, y_train)
preds_test = model.predict(final_X_test)

# Check your answers
step_4.b.check()
```