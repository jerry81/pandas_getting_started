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