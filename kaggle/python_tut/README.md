# kaggle [python](https://www.kaggle.com/learn/python) course

## hello python

- you can do mult on a string and i guess it duplicates the string x times
- type(var_name) - comes out of the box
- division: "/" true division, "//" floor division
- exp is "a ** b"
- type conversion
```py
float(1) # 1.0
int(1.0) # 1
int("807") + 1 # 808
```

- swap values of variables sugar!
```py
a, b = b, a
```

## functions and getting help

- help() function most important fn you can learn
- gets the api docs

- cool things with print
```py
print(1,2,3 sep=' < ')
```

- to add default values
```py
def greet(who="Colin"):
    print("Hello " + who)
```

## lists

- get index of a item in list
- l.index("item")
- if item doesn't exist, it throws an error

## loops
- looping lists
```py
  for item in list:
    print(item)
```
- looping with index
```py
  for i in range(len(list)):
    print(i, list[i])
```
- TIL: list comprehension
```py
short_planets = [planet for planet in planets if len(planet) < 6]
```
- like "where" in SQL
- list comprehension like Select from where in sql
- the above compreshension is equivalent to
```py
short_planets = []
for planet in planets:
    if len(planet) < 6:
        short_planets.append(planet)
```
- TIL: "any"
- the following returns true if any of the numbers satisfies the condition
```py
def has_lucky_number(nums):
    return any([num % 7 == 0 for num in nums])
```

## strings and dicts
- triple quotes like markdown
- print("", end="") - to avoid new line
- join syntax
```py
"separator".join(list_of_strings)
```
- format syntax
```py
"{} is a {}".format("Earth", "planet")
```
- curly brackets to declare dictionaries
- access with [], so same syntax as javascript
- "in" to check if key exists in dict
```py
planets = {"Mercury": 0.39, "Venus": 0.72, "Earth": 1.00, "Mars": 1.52}
"Earth" in planets # True
```
- keys() and values() to get collection of keys and values
- items() to get collection of key value pairs