[source](https://www.kaggle.com/competitions/titanic)

# titanic competition
- this is a getting started competition

- use ML to create model that predicts which passengers survived the Titanic shipwreck

## tutorial from alexis cook
- created in 2022
1.  join competition - click "join competition" button
2.  view 3 files
  - train - 891 passengers, 1 psgr per row
  - test - 418 passengers - survived column left blank
  - gender submission - shows example of prediction that all females passengers survived
3.  train ML model, write code in kaggle notebook
4.  boilerplate code - import numpy and pandas, read in data
5.  explore data - df.loc[Sex=='female']["Survived"] - gets rows that match the filter - by sum(women)/len(women) finds % of women who survived
6.  build random forest model
  - construct 100 trees, individually consider each passengers' data and vote on whether individual survived
  - randomForestClassifier imported from sklearn.ensemble
7.  output to csv with 2 columns
  - passengerId
  - survived

## TILs

- get_dummies() - converts categorical values into numeric indicator columns (one-hot encoding)

- scikit-learn (sklearn) - the go-to py lib for classical ML
  - preprocessing
  - models - linear, logistic regression, decision trees, random forests, SVM, k-NN, clustering
  - evaluation - accuracy, precision/recall, ROC-AUC, confusion matrix, cross-validation
  - pipelines - chain preprocessing + model
  - model tuning

- random forest - ensemble method that constructs multiple decision trees and outputs the mode of their predictions
  - called ensemble method because it combines predictions from multiple models to improve accuracy and reduce overfitting
  - 2 random aspects - random rows picked to construct tree, trees deliberately randomized to be different from e/o
  - each level, a column is chosen and one threshold or value is split on
    - sexM 1, age < 14 etc
  - each node ends up with a yes/no match, so boolean
  - always produces prediction tree for each row
  - binary tree
  - each tree only contains a random subset of columns, capturing a slightly different "view" of what predicts survival