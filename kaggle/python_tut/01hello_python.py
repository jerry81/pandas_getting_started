spam_amount = 0
print(spam_amount)

# Ordering Spam, egg, Spam, Spam, bacon and Spam (4 more servings of Spam)
spam_amount = spam_amount + 4

if spam_amount > 0:
    print("But I don't want ANY spam!")

viking_song = "Spam " * spam_amount
print(viking_song)

##### this

a = [1, 2, 3]
b = [3, 2, 1]

c = a
a = b
b = c

##### is equivalent to

a,b = b,a

print(1,2,3, sep=' < ')

help(round)

### conditionals ###
### til conditional assignment

total_candies = 1

print("Splitting", total_candies, "candy" if total_candies == 1 else "candies")