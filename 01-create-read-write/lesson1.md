# step 1 - import the library

import pandas as pd

# two core objects

- dataframe
  - a table
  - strings or integers
  - constructor, pd.DataFrame()
  - index -> creates labels for columns
- series
  - sequence of values

# those objects typically created by reading files

- usually csv
- importedObj = pd.read_csv("../path/to/file.csv")

- check size after read in
- importedObj.shape

- preview first 5 records
- importedObj.head()