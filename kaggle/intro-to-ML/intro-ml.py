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
print("The predictions are")
print(melbourne_model.predict(X.head()))

# dt vs rf
# dt is a single tree, rf is colleciton of trees with random subset of features