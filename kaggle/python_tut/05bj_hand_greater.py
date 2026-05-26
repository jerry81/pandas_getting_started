def calc_highest_hand(hand):
    handtotal = 0
    acecount = 0
    for item in hand:
        if item == 'A':
            acecount += 1
        elif item in ['J', 'Q', 'K']:
            handtotal += 10
        else:
            handtotal += int(item)

    handtotal += acecount
    if acecount > 0 and handtotal + 10 <= 21:
        handtotal += 10

    return handtotal

def blackjack_hand_greater_than(hand_1, hand_2):
    hand1total = calc_highest_hand(hand_1)
    hand2total = calc_highest_hand(hand_2)

    if hand1total > 21:
        return False
    if hand2total > 21:
        return True

    return hand1total > hand2total