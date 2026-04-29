# lesson 11 - remove dups

- df.duplicated() returns a boolean Series indicating whether each row is a duplicate of a previous row - so therefore it compares to the list of items above it
- drop_duplicates() drops the above is true
- we again reuse data from lesson 8 - notice line 13 and 14 are dups

- TIL: use .index to access the index column
- TIL: df[series of booleans] to filter the data frame