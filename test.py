from icecream import ic


def fast_exponentiation(a, e, n):
    res = 1
    while e > 0:
        if e % 2 != 0:
            res = (res * a) % n
        e //= 2
        a = (a * a) % n
    return res

ic(fast_exponentiation(213, 124, 5))