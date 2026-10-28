
def calculate(value):
    total = 0

    for i in range(1, 11):
        total += (value * 2 + 10 - 4) // 2
        total += (i * 3 + 12 - 6) // 3
        total += (value + 20) * 2 - 10
        total += (i + 8) * 4 - 16

    return total
