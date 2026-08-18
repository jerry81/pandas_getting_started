# [Kaggle Data Cleaning](https://www.kaggle.com/learn/data-cleaning)

## Handling Missing Values

- Look at data: Events in American football games
  - GameID, Date, Drive, qtr, down, time, TimeUnder, TimeSecs, PlayTimeDiff, SideofField ... yacEPA, Home_WP_pre
  - use head() to inspect the data
  - some missing values present NaN
  - check num of missing data points
```py
missing_values = df.isnull().sum()
# display first 10 cols' numbers
missing_values[0:10]
# calc percent missing
total_cells = np.product(nfl_data.shape)
total_missing = missing_values.sum()
percent_missing = (total_missing/total_cells) * 100
```
  - found ~25 percent missing

- figuring out why data is missing
  - "data intuition" - realling looking at data and finding out why it is the way it is, and how it will affect analysis
  - question 1: is data missing because it wasn't recorded, or because it didn't exist
    - if it didn't exist, then that's fine, leave as NaN.
    - if it did exist, then can fill in with guess aka imputation
  - first we examine column with many missing values
    - look at documentation for data set for more info
      - tip: if no documentation, then reach out to person who provided data for more info
    - found out missing because not recorded, so should try to fill in with imputation

- drop missing data
  - option if you're in a hurry
  - dropna()
```py
nfl_data.dropna()
```
  - but a caveat
    - drops rows with one or more missing values
  - remove columns with at least 1 missing value
```py
nfl_data.dropna(axis=1)
```
  - use axis=1 to drop columns, axis=0 to drop rows

- fill missing values with constant
  - fillna()
```py
nfl_data.fillna(0)
```