
## intro to machine learning
[ref](https://www.kaggle.com/learn/intro-to-machine-learning)
- predicting housing prices
- decision tree
  - boolean test i.e. "does house have more than 2 bedrooms"

## your first machine learning model
- pare down data to something understandable
- df.columns - see list of all columns
- using sklearn for traditional models
- steps - find target column (the column you predict) aka y
- find your selector for the model (a bunch of columns) X
- initialize model
- fit(X,y)
- model.predict(some rows with subset of columns)

## model validation
- this step is to improve models iteratively
- question: how to measure model quality (predictive accuracy)
  - error = actual - predicted
  - MAE - mean absolute error - average of all the errors in a set of rows
- reminder - y is actual values, predicted is .predict(x) results
```
mean_absolute_error(y, predicted_home_prices)
```
- key concept for things like decision tree - try to use different data for testing error than training data
- use train_test_split to acheive this
```
train_X, val_x, train_y, val_y = train_test_split(X, y, random_state=0)
```

- then you use train_x, train_y to fit
- then predictions use val_x
- finally, mean_absolute_error use result of val_x predictions and val_y

## underfitting, overfitting

- decision tree - many options can be put in - including tree depth
- in practice common to have 10 splits
- overfitting - when a leaf in the tree has too few examples, it will not do well with new data, and will contain data too close to training data

- if too few splits are in the tree, groups have huge variety of houses, resulting in underfitting - may even perform poorly in training data

- so to find the best balance, test mean absolute error, experiment with max_leaf_nodes as input

- function to find mae based on max nodes as input
```py
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
```

