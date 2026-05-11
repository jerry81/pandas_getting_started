import pandas as pd
from sklearn.tree import DecisionTreeRegressor

melbourne_file_path = './data/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)
print(melbourne_data.describe())
# til: describe() method gives us a quick overview of the numerical data in our DataFrame.

print(melbourne_data.columns)

melbourne_data = melbourne_data.dropna(axis=0)

# convention - prediction column called y
y = melbourne_data.Price

# "features" are columns you input into model
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = melbourne_data[melbourne_features]

# random_state ensures same result each run
melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X,y)

# in decision tree

print("Making predictions for the following 5 houses:")
print(X.head())
print("whose prices are")
print(y.head())
print("The predictions are")
# predict takes in X, which are the features, and outputs the predicted target column values
print(melbourne_model.predict(X.head()))

# dt vs rf
# dt is a single tree, rf is colleciton of trees with random subset of features

####### part 2: validation #######

from sklearn.metrics import mean_absolute_error
print("mean using same train/test rows")
train_predictions = melbourne_model.predict(X)
print(mean_absolute_error(y, train_predictions))

train_errors = (y - train_predictions).abs()
print("max absolute error on training rows")
print(train_errors.max())
print("number of rows with non-zero error")
print((train_errors > 0).sum())
print("first rows where prediction differs from actual")
debug_df = pd.DataFrame({
	'actual': y,
	'predicted': train_predictions,
	'abs_error': train_errors,
})
print(debug_df[debug_df['abs_error'] > 0].head())

from sklearn.model_selection import train_test_split


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
melbourne_model.fit(train_X, train_y)
val_predictions = melbourne_model.predict(val_X)
print("mean using split test/train")
print(mean_absolute_error(val_y, val_predictions))

####### part 3: under and overfitting #######

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
import sys
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
best_size = 5
mx = sys.float_info.max
for mx_ln in candidate_max_leaf_nodes:
    mae =  get_mae(mx_ln, train_X, val_X,train_y, val_y)
    if mae < mx:
        best_size = mx_ln
        mx = mae


# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = best_size
