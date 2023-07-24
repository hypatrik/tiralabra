def assertAlmostEqual(x, y, digits=7):
    multiplier = 10 ** digits
    assert round(x * multiplier) == round(y * multiplier), f"{x} and {y} are not almost equal."
