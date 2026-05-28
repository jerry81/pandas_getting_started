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