# lesson 2 - series
- series is one dimensional array
- created from list
- pd.Series([1,2,3])

- create labels with index
- pd.Series([1,2,3], index==['a','b','c'])

- series from dict
```
calories = {"day1": 420, "day2": 380, "day3": 390}
pd.Series(calories)
```
- series from part of a dict
```
calories = {"day1": 420, "day2": 380, "day3": 390}

myvar = pd.Series(calories, index = ["day1", "day2"])
```