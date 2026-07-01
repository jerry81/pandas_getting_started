[from this link](https://www.kaggle.com/learn/pandas)

# crete, read, write
- index -> row label
```py
pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'],
              'Sue': ['Pretty good.', 'Bland.']},
             index=['Product A', 'Product B'])
```

- to read a csv using column 0 from csv, use index_col=0
```py
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v1.csv", index_col=0)
```

- to write a csv, use to_csv
```py
reviews.to_csv("winemag-data-130k-v1.csv")
```

# index, select, assign
- sq brackets to access keys in dict, same with DF
- iloc method: position based selection
- ':' means everything
```py
reviews.iloc[:,0]
```
- means get the first column


- the following means get the first 3 rows
```py
reviews.iloc[[0, 1, 2], 0]
```

- conditional selection
```py
reviews.country == 'Italy'
```
- selects all rows, with an extra column (bool) indicating if country column value is "Italy"

- single ampersand operator brings two conditionals together

- isin selector: like js includes, supply a list

- isnull/notnull

- filter by multiple conditionals
```python
top_oceania_wines = reviews.loc[((reviews.country == 'Australia') | (reviews.country == 'New Zealand')) & (reviews.points >= 95)]
```

# summary functions and maps

- describe() - high level summary of column
- mean()
- unique() - list of unique vals
- value_counts() - count of unique vals
- head(num_of_rows) - first num_of_rows rows
- map(fn) - can use lambda
  - lambda syntax
```
lambda p: p - review_points_mean
```
- apply(fn, axis='index|columns') - transform DF, columnns - transforms each row,
index -> transforms each column