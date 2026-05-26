def blackjack_hand_greater_than(hand_1, hand_2):
    hand1total = 0
    acecount = 0
    for item in hand_1:
        if item == 'A':
            acecount+=1
        elif item in ['J', 'Q', 'K']:
            hand1total+=10
        else:
            hand1total+=item
    print("hand1total: ", hand1total)
    pass