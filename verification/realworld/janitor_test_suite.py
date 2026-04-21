def test_calc_positive():
    assert calc(1, 2) == 3

def test_calc_zero_and_negative():
    assert calc(0, 5) == 5
    assert calc(-1, 10) == 10
