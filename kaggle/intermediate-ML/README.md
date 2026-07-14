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

- quick review
- first step: identify inputs and output
- inputs - one ore more columns - (size + bedrooms + age)
- output - one column (price)
- inputs = X = features
- output = y = target
- train_x, train_y splits training data into the features and the target
- fit() - given ML model, call fit pass in train_x and train_y to "learn" from the data - in case of RF, actually builds the forest
- predict() - given trained ML model, predict runs the model on new data to make predictions
- valid_x - could be the original data split into training and validation
- predict called on valid_x results compared to valid_y
- test_x test_y -> same as valid, but test_y is not known, so we cannot compare predictions to actual values

- steps in order
1.  load data
2.  split into train and valid
3.  create model
4.  fit with trainX, trainY
5.  prdedict with validX
6.  compare predictions to validY, i.e. mean_absolute_error(validY, predictions)

# categorical Variables

- takes limited num of values (like enum?)
- problem - these variables give error without preprocessing first
- dealing with Categorical data
  1. drop directly
  2.  ordinal encoding
    - assign each "value" a number
    - better when there is clear ordering
  3.  one-hot encoding
    - create a new column for each value, with 1 or 0 indicating presence of that value
    - caveat: bad for when there exist a large number of possible values
- tutorial shows how to identify categorical variables: check type, look for type "object"
  - in pandas string columns end up with column type object

- impl of method 1
```
drop_X_train = X_train.select_dtypes(exclude=['object'])
```
- impl of method 2: needs scikit-learn
```py
from sklearn.preprocessing import OrdinalEncoder

# Make copy to avoid changing original data
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])
```
- impl of method 3: also uses scikit-learn

```py
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Ensure all columns have string type
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

```
q. what's the difference between ordinal_encoder.fit_transform and transform?

a.  the fit refers to learning what possible values need to be encoded, and the transform refers to actually doing the encoding.  The fit_transform does both in one step, while transform only does the encoding based on a previously fitted encoder.
a. given the block

label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

the first line is just doing the fit first and then transform the second line since the model is already trained, can just do transform alone both X_... dataframes are transformed the same
```

- note that method 3 requires dropping original CV columns and inserting the encoded ones back
- method 2, since it only involves a single column, replaces the column