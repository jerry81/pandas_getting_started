def should_hit(dealer_total, player_total, player_low_aces, player_high_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay.
    When calculating a hand's total value, we count aces as "high" (with value 11) if doing so
    doesn't bring the total above 21, otherwise we count them as low (with value 1).
    For example, if the player's hand is {A, A, A, 7}, we will count it as 11 + 1 + 1 + 7,
    and therefore set player_total=20, player_low_aces=2, player_high_aces=1.
    """
    # In this simulator, dealer shows one card and ties are dealer wins.
    # These thresholds are the exact win-rate-optimal policy for that ruleset.

    # Soft hands (at least one ace counted as 11).
    if player_high_aces > 0:
        if player_total <= 17:
            return True
        if player_total == 18:
            return dealer_total >= 8
        return False

    # Hard hands.
    if dealer_total <= 6:
        return player_total <= 11
    if dealer_total == 7:
        return player_total <= 16
    if dealer_total in (8, 9):
        return player_total <= 15
    if dealer_total == 10:
        return player_total <= 14
    # Dealer ace (11): hit through 17 because ties lose.
    return player_total <= 17

    return False
# q7.simulate(n_games=50000)