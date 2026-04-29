# lesson 8 - clean empty cells

- empty cells may give wrong results when analyzing
- use df.dropena() to remove rows with empty cells
- inplace=True flag to keep original data frame, rather than creating a new one
- fillna(value_to_fill, inplaceflag) to replace empty cells

- common practice - remove empty with average