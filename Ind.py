import matplotlib.pyplot as plt
import numpy as np


def f1(x):
    return abs(x) / 2


def f2(x):
    return int(abs(x) / 2)


x = np.arange(-10, 10, 0.1)
y1 = np.vectorize(f1)
y2 = np.vectorize(f2)
plt.figure(1)
plt.plot(x, y1(x), color='blue', label='f(x) = |x| / 2')
plt.plot(x, y2(x), color='red', label='f(x) = [x] / 2')
plt.grid(True)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
print(f'N1 in plots')
# ||| 2 |||

from scipy.optimize import fsolve


# Определяем функцию, которую хотим решить
def equation(x):
    # Возвращаем разницу между левой и правой сторонами исходного уравнения
    return (2 - abs(5 * x)) / 3 - (x - int(x))


# Начальное приближение для x
initial_guess = [0]

# Поиск решения
solution = fsolve(equation, initial_guess)

print(f'N2 -> Решение уравнения: x приблизительно равно {solution[0]}')

# ||| 3 |||
# Выведем все положительные делители числа 135
deliteli = [i for i in range(1, 136) if 135 % i == 0]
print(f"N3 -> {deliteli}")


# ||| 4 |||
# Предполагаем, что функция линейна: f(x) = k * x
def f(x, k=1):
    return k * x


res_count = []

# Проверяем уравнение для всех значений x в интервале [15, 25] с шагом 1
for x in range(15, 26):  # Включительно 25, поэтому 26 не включается
    if f(125 * x) == 5 * f(x):
        res_count.append(f(125 * x))
        print(f"N4 -> Уравнение выполняется для x = {x}")
if len(res_count) == 0:
    print(f"N4 -> Уравнение не выполняется")


# ||| 5 |||


# ||| 6 |||


# ||| 7 |||
def find_solution():
    n = 9  # число n
    result = -1  # инициализируем результат -1, если нет решения
    for x in range(1, n):  # перебираем значения x от 1 до n-1
        if (3 * x) % n == 6 % n:  # проверяем условие уравнения
            result = x  # сохраняем значение x, если условие выполняется
            break  # выходим из цикла, если нашли решение

    return result


solution = find_solution()
print(f"N7 решение -> {solution}")  # выводим решение


# ||| 8 |||
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a


def prime_factors(n):
    factors = {}
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            if i in factors:
                factors[i] += 1
            else:
                factors[i] = 1
    if n > 1:
        if n in factors:
            factors[n] += 1
        else:
            factors[n] = 1
    return factors


def find_values():
    result = []  # инициализируем список для хранения значений n
    for n in range(2, 100):  # проверяем значения n от 2 до 100
        prime_factors_n = prime_factors(n)
        # Вычисляем произведение основных простых множителей для чисел 17 и 16
        product_17 = 1
        product_16 = 1
        for prime, power in prime_factors_n.items():
            product_17 *= prime ** power if prime != 2 else prime ** (power - 1)
            product_16 *= prime ** power

        if (17 % product_17 == 16 % product_16 and
                gcd(product_17, n) == 1 and
                gcd(product_16, n) == 1):
            result.append(n)
    return result


values = find_values()
if len(values) == 0:
    print(f"N8 -> Нет решения")
else:
    print(f"N8 -> {values}")  # выводим значения n, при которых выполняется равенство


# ||| 9 |||

def check_congruence(a, b, n, k, d):
    assert d > 1, "d должно быть больше 1"
    assert k % d == 0, "k должно быть кратно d"

    left_side = (k * a) % (d * n)
    right_side = (k * b) % (d * n)

    is_congruent = left_side == right_side
    return is_congruent


# Пример использования:
a = 5
b = 2
n = 3
k = 4  # Например, k может быть кратным числа d
d = 2  # d должно делить k и быть больше 1

# Проверяем утверждение
result = check_congruence(a, b, n, k, d)
print(f"N9 -> Утверждение о равенстве {'верно' if result else 'неверно'}")

# ||| 10 |||
base = -210
s1 = 5
s2 = 2010
print(f"N10 -> {pow(base, s1 * s2) % 208}")

# ||| 11 |||
from sympy import symbols, Eq, solve

# Создаем символьную переменную
x = symbols('x')

# Задаем уравнения системы сравнений
eq1 = Eq(11 * x, -2)
eq2 = Eq(10 * x, 6)
eq3 = Eq(12 * x, 15)

# Решаем систему сравнений
solutions = solve((eq1, eq2, eq3), x)
# Выводим решения
for solution in solutions:
    print(f"x ≡ {solution} (mod 6), {solution} (mod 14), {solution} (mod 27)")

if len(solutions) == 0:
    print(f"N11 -> Нет решений")

# ||| 12 |||
# чему равны остатки от деления коэффициентов и показателей степени на 7:
#
# 452 ≡ 6 (mod 7) так как 452 делится на 7 с остатком 6.
# 50 ≡ 1 (mod 7) так как 50 делится на 7 с остатком 1.
# 82 ≡ 3 (mod 7) так как 82 делится на 7 с остатком 3.
# 226 ≡ 3 (mod 7) так как 226 делится на 7 с остатком 3.
# 102 ≡ 3 (mod 7) так как 102 делится на 7 с остатком 3.
for x in range(7):
    equation = (4 * x ** 4 - 3 * x ** 5 + 4) % 7
    if equation == 0:
        print(f"N12 -> x = {x} является решением уравнения.")

# ||| 13 |||
from sympy import legendre_symbol

legendre_symbol(-288, 461) # ;/
print(f"N13 -> {legendre_symbol(-288, 461)}")

# ||| 14 |||
from sympy import symbols, Eq, solve

x = symbols('x')
equation = Eq(x**2, 22)
solutions = solve(equation, x, domain=[0, 37-1])
print(f'N14 кол-во решений -> {len(solutions)}')

# ||| 15 |||
# Подключаем модуль sympy
from sympy import symbols, solve, mod_inverse

# Определяем переменную x
x = symbols('x', integer=True)

# Определяем сравнение: x**2 + 5*x + 3 ≡ 0 (mod 43)
equation = x**2 + 5*x + 3

# Находим все решения в пределах от 0 до 42 (набор вычетов по модулю 43)
solutions = [sol.evalf() for sol in solve(equation, x) if sol.is_Integer and 0 <= sol.evalf() < 43]

print(f'N15 кол-во решений -> {len(solutions)}')

# ||| 16 |||
# Определим модуль
modulus = 453

# Инициализируем переменную для счетчика решений
solution_count = 0

# Перебираем все возможные значения x в пределах от 0 до modulus-1
for x in range(modulus):
    # Вычисляем значение квадратного выражения по модулю modulus
    if (51*x**2 + x + 23) % modulus == 0:
        # Если условие выполнено, увеличиваем счетчик решений
        solution_count += 1

print(f"N16 -> Количество решений сравнения: {solution_count}")

# ||| 17 |||
from sympy import prime
prime_73 = prime(73)
result = prime_73 ** 49

print(f"N17 -> P₇₃(49) или 73-е простое число в степени 49: {result}")
# ||| 18 |||
from sympy import prime
prime = prime(53*pow(33, 5))
result = prime ** 10
print(f"N18 -> {result}")
# ||| 19 |||
def find_period(numerator, denominator):
    remainder = numerator % denominator
    remainder_list = []

    while remainder != 0 and remainder not in remainder_list:
        remainder_list.append(remainder)
        remainder *= 10
        remainder = remainder % denominator

    if remainder == 0:
        return 0  # дробь не имеет периода

    return len(remainder_list) - remainder_list.index(remainder)
numerator = 2
denominator = 14

period_length = find_period(numerator, denominator)
print('N19 ->', period_length)

# ||| 20 |||
def find_period(numerator, denominator, base):
    remainder = numerator % denominator
    remainders = []

    while remainder not in remainders:
        remainders.append(remainder)
        remainder = (remainder * base) % denominator
        if len(remainders) > 100:
            return None

    return len(remainders)

numerator = 85
denominator = 23 * pow(30, 6)
base = 14

period_length = find_period(numerator, denominator, base)
if period_length is None:
    print("N20 -> Дробь не имеет периода или его длина больше 100.")
else:
    print(f"N20 -> Длина периода g-ичной записи дроби:{period_length}")

# ||| 21 |||
# Находим обратный элемент для коэффициента 155 по модулю 89
a_inv = pow(155, -1, 89)

# Находим решение уравнения в виде x^40 ≡ b (mod 89), где b - это произведение обратного элемента коэффициента и правой части сравнения
b = (a_inv * 33) % 89

# Возведение в степень по модулю можно упростить,
# используя малую теорему Ферма. Поскольку p=89 простое число,
# a^(p-1) ≡ 1 (mod p), сокращаем степень 40 до 40 mod 88.
exp = 40 % (89 - 1)

# Теперь находим x^exp ≡ b (mod 89)
x = pow(b, exp, 89)

print(f"N21 -> {x}")

# ||| 22 |||

# ||| 23 |||
def is_primitive_root(a, p):
    if pow(a, p - 1, p) != 1:
        return False
    factors = factorization(p - 1)
    for factor in factors:
        if pow(a, (p - 1) // factor, p) == 1:
            return False
    return True

def factorization(n):
    factors = []
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            factors.append(i)
            while n % i == 0:
                n //= i
    if n > 1:
        factors.append(n)
    return factors

p = 47
range_start = 4
range_end = 14

primitive_roots = []
for a in range(range_start, range_end + 1):
    if is_primitive_root(a, p):
        primitive_roots.append(a)

print(f" N23 -> {primitive_roots}")
# ||| 24 |||
def quadratic_nonresidues(modulus, start, end):
    nonresidues = []
    for x in range(start, end + 1):
        is_nonresidue = True
        for y in range(1, modulus):
            if (y ** 2) % modulus == x:
                is_nonresidue = False
                break
        if is_nonresidue:
            nonresidues.append(x)
    return nonresidues

modulus = 41
start = 1
end = 9
nonresidues = quadratic_nonresidues(modulus, start, end)
print("N24 -> Квадратичные невычеты по модулю 41 на отрезке [1, 9]:", nonresidues)

# ||| 25 |||
print(f"N25 -> {pow(34, 48*15)%59}")

# ||| 26 ||
def find_indexes(numbers, target):
    indexes = []
    for i, number in enumerate(numbers):
        if number % target == 0:
            indexes.append(i)
    return indexes

numbers = [23, 5, 12, 23, 9, 23, 17]
target = 19

indexes = find_indexes(numbers, target)
print(f"N26 -> {indexes}")

# ||| 29 |||
from sympy import symbols, Eq, solve

# Объявление переменных
x, y = symbols('x y')

# Уравнение
equation = Eq(618*y + 1050*x, 6)

# Решение уравнения
solution = solve(equation, (x, y))

# Вывод результата
print(f"N29 -> {solution}")