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

- idxmax() - returns index of max value in column
- "vectorized" arithmetic
```
ratio = reviews["points"] / reviews["price"]
```
- .str - special pandas accessor - apply string operations
- str.contains()

# grouping and sorting

- groupby() - split-apply-combine
```
reviews.groupby('points').price.min()
```
- this snippet groups the reviews by points, and then finds the minimum price for each group
```
reviews.groupby(['country', 'province']).apply(lambda df: df.title.iloc[0])
```
- this snippet groups the reviews by country and province, and then applies a function to each group
- results in a multi-index
- multi-index must be indexed by a tuple of values, in the order of the groupby columns
- multi-index can be reset to a normal index using reset_index()
```
reviews.groupby('country').price.agg([len, min, max])
```
- this snippet groups the reviews by country, and then applies multiple aggregation functions to the price column
- agg() - can take a list of functions, or a dict of column names to functions
- .sort_values(by='column_name', ascending=True|False) - sorts the DF by the specified column
- sort_index() - sorts the DF by the index
```
countries_reviewed.sort_values(by=['country', 'province'], ascending=[True, False])
```
- this snippet sorts the countries_reviewed DF by country in ascending order, and then by province in descending order