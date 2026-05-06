# what is Kaggle

- Kaggle is essentially a platform for learning and practicing data science and machine learning using real datasets.

## features

- hosts tones of datasets
- images for cv
- text NLP

- training and competition

## first steps

- go to kaggle
- open titanic dataset
- df.head()
- df.info ()

## about the data set, titanic

- sibsp - siblings spouses
- porch - parents/children
- pclass.  passenger class
- zero: one-hot encoded columns
  - takes categorical column and explodes it to many binary columns
  - way to convert categorical data (labels + words) into numbers so ML can use them
  - removes ordering
  - categories independent
  - instead of
```
red=1
blue=2
green=3
```
  - we have 3 columns
```
red=1,0,0
blue=0,1,0
green=0,0,1
```