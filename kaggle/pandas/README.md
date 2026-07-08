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

## exercise

```
reviews_written = reviews.groupby("taster_twitter_handle").taster_twitter_handle.count()
```
- notice group_by returns object of type DataFrameGroupBy, which is not a DF, but can be transformed into one using aggregation functions like count(), mean(), etc.
- the dataframegroupby offers accessors to columns

```
best_rating_per_price = reviews.groupby("price").points.max()
```
- this snippet groups the reviews by price, and then finds the maximum points for each group


- what does "'method' object is not subscriptable mean"?
- means you used square brackets instead of round
```py
country_variety_counts = reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)
```
- size() used to get the size of each group, returns a series with a multi-index


# data types and missing values

- dtypes - returns a series with the data types of each column
- converting column data type
```py
reviews.points.astype('float64')
```

## finding missing values

- the following gets the rows where a column has missing values
```py
reviews[pd.isnull(reviews.country)]
```
- to replace missing values
```py
reviews.country.fillna("Unknown", inplace=True)
```
- to replace non-null value
```py
reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")
```

## exercise

- got error
```

ValueError: The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().

```
- this error occurs when you try to use a DataFrame in a boolean context, such as an if statement or a logical operation. A DataFrame can have multiple rows and columns, so it is not clear what the truth value of the entire DataFrame should be. To resolve this, you can use methods like .empty, .any(), or .all() to check specific conditions on the DataFrame.

- so we use .sum() to count the number of True values in the boolean DataFrame, which gives us a single integer value that can be used in a boolean context.
- this
```
n_missing_prices = reviews.price.isnull().sum()
```
- is equivalent to
```
n_missing_prices = len(reviews[pd.isnull(reviews.price)])
```

- the following snippet counts the number of reviews per region, filling in missing values with 'Unknown' before counting
```
reviews_per_region = reviews.region_1.fillna('Unknown').value_counts().sort_values(ascending=False)
```

- explanation
```
reviews is a DataFrame.
reviews.region_1 picks one column from that DataFrame.
A single column is a Series.
.fillna("Unknown") keeps it a Series.
.value_counts() returns a new Series:
values = counts
index = unique region names from region_1 (plus "Unknown" if filled)
```

# renaming and combining

- syntax to rename column(s)
```
reviews.rename(columns={'points': 'score', 'price': 'cost'}, inplace=True)
```
- can also rename default indexes
```
reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})
```
- rename entire axises
```
reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')
```

## combining

- you can combine two dataframes using concat(), join() <-- used when common index is available (also multiindex)
- concat syntax
```py
canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")
british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")

pd.concat([canadian_youtube, british_youtube])
```
- lsuffix and rsuffix can be used to avoid column name conflicts

```py
left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])

left.join(right, lsuffix='_CAN', rsuffix='_UK')
```