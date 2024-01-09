import random
from sympy import *

while True:
    command = input("Выберете лабараторную работу [1, 2, 3, 4 (4 пункт 5 и 4 лабараторная)]: ")
    match command.split():
        case ['1']:
            command = input("Выберете задание [1, 2, 3]: ")
            match command.split():
                case ['1']:
                    def l1_1(a, e, n):
                        res = 1
                        while e > 0:
                            if e % 2 != 0:
                                res = (res * a) % n
                            e //= 2
                            a = (a * a) % n
                        return res
                    a = int(input("Введите число a: "))
                    e = int(input("Введите число e: "))
                    n = int(input("Введите модуль n: "))
                    result = l1_1(a, e, n)
                    print("Вычет a^e по модулю n:", result)
                    print(a**e%n)
                case ['2']:
                    def l1_2(a, b):
                        while b:
                            a, b = b, a % b
                        return a
                    a = int(input("Введите первое число (a): "))
                    b = int(input("Введите второе число (b): "))
                    result = l1_2(a, b)
                    print(f"Наибольший общий делитель {a} и {b} равен {result}")

                case ['3']:
                    def l1_3(a, b):
                        result = []
                        while b != 0:
                            q = a // b
                            result.append(q)
                            a, b = b, a % b
                        return result
                    a = int(input("Введите числитель (a): "))
                    b = int(input("Введите знаменатель (b): "))
                    result_list = l1_3(a, b)
                    print(f"Неполные частные дроби для {a}/{b}: {result_list}")

                case _:
                    print("Такого задания нет")

        case ['2']:
            command = input("Выберете задание [1, 2, 3]: ")
            match command.split():
                case ['1']:
                    def l2_1(n):
                        is_prime = [True] * (n + 1)
                        is_prime[0] = is_prime[1] = False
                        for i in range(2, int(n**0.5) + 1):
                            if is_prime[i]:
                                for j in range(i*i, n+1, i):
                                    is_prime[j] = False
                        primes = [i for i in range(2, n+1) if is_prime[i]]
                        return primes

                    n = int(input("Введите натуральное число n >= 2: "))
                    if n < 2:
                        print("n < 2")
                    else:
                        print(f"Простые числа от 1 до {n}: {l2_1(n)}")

                case ['2']:
                    def fast_power(base, exponent, mod):
                        result = 1
                        base = base % mod
                        while exponent > 0:
                            if exponent % 2 == 1:
                                result = (result * base) % mod
                            base = (base * base) % mod
                            exponent //= 2
                        return result

                    def fermat_test(n, a):
                        if not (isinstance(n, int) and isinstance(a, int) and n > 0 and a > 0):
                            raise ValueError("Числа должны принадлежать множеству натуральных чисел")
                        result = fast_power(a, n - 1, n)
                        return result == 1

                    n = int(input("Введите натуральное число n: "))
                    a = int(input("Введите натуральное число a: "))
                    result = fermat_test(n, a)
                    print(result)

                case ['3']:
                    def is_prime(n, k=5):
                        if n == 2 or n == 3:
                            return True
                        if n == 1 or n % 2 == 0:
                            return False

                        s, d = 0, n - 1
                        while d % 2 == 0:
                            s += 1
                            d //= 2

                        for i in range(k):
                            a = random.randint(2, n - 2)
                            x = pow(a, d, n)
                            if x == 1 or x == n - 1:
                                continue
                            for i in range(s - 1):
                                x = pow(x, 2, n)
                                if x == n - 1:
                                    break
                            else:
                                return False
                        return True
                    def find_pseudoprimes(l, r):
                        pseudoprimes = []
                        for p in range(l, r + 1):
                            if all(p % prime != 0 for prime in range(2, 5000)) and \
                                    any(is_prime(p, base) for base in [2, 3, 5, 7, 11]):
                                pseudoprimes.append(p)
                        return pseudoprimes

                    l = int(input("Введите натуральное число l: "))
                    r = int(input("Введите натуральное число r: "))
                    result = find_pseudoprimes(l, r)
                    print(f"Простые псевдопростые числа на отрезке [{l}, {r}]: {result}")

        case ['3']:
            command = input("Выберете задание [1]: ")
            match command.split():
                case ['1']:
                    def egcd(a, b):
                        if a == 0:
                            return b, 0, 1
                        else:
                            g, x, y = egcd(b % a, a)
                            return g, y - (b // a) * x, x

                    def modinv(a, m):
                        g, x, y = egcd(a, m)
                        if g != 1:
                            raise Exception('Обратного элемента не существует')
                        else:
                            return x % m

                    def gcd(a, b):
                        while b:
                            a, b = b, a % b
                        return a

                    def continued_fraction(a, m):
                        result = []
                        while m != 0:
                            result.append(a // m)
                            a, m = m, a % m
                        return result

                    def fast_power(base, exponent, mod):
                        result = 1
                        base = base % mod

                        while exponent > 0:
                            if exponent % 2 == 1:
                                result = (result * base) % mod

                            base = (base * base) % mod
                            exponent //= 2

                        return result

                    def linear_congruence_solver(a, b, m):
                        if gcd(a, m) != 1:
                            raise ValueError("a и m должны быть взаимно простыми")

                        a_inv = modinv(a, m)
                        cf_expansion = continued_fraction(a, m)
                        x0, x1 = cf_expansion[0], cf_expansion[0] * cf_expansion[1] + 1
                        x = fast_power(a, b, m) * a_inv % m
                        solutions = [x % m]

                        for i in range(2, len(cf_expansion) + 1):
                            x0, x1 = x1, cf_expansion[i - 1] * x1 + x0

                        for k in range(1, len(cf_expansion)):
                            x = cf_expansion[k] * solutions[k - 1] + x0
                            solutions.append(x % m)

                        return solutions
                    a = int(input("Введите натуральное число a: "))
                    b = int(input("Введите натуральное число b: "))
                    m = int(input("Введите натуральное число m: "))
                    # линейное сравение
                    try:
                        solutions = linear_congruence_solver(a, b, m)
                        print("Решения линейного сравнения {}x ≡ {} (mod {}):".format(a, b, m))
                        if solutions:
                            print("Вычеты x: {}".format(solutions))
                        else:
                            print("Решений нет")
                    except ValueError as e:
                        print(e)
                    except Exception as e:
                        print("Произошла ошибка:", e)
                    def chinese_remainder_theorem(b1, b2, m1, m2):
                        if egcd(m1, m2)[0] != 1:
                            raise ValueError("Модули m1 и m2 должны быть взаимно простыми")
                        m1_inv = modinv(m1, m2)
                        m2_inv = modinv(m2, m1)
                        x = (b1 * m2 * m2_inv + b2 * m1 * m1_inv) % (m1 * m2)
                        return x
                    b1 = int(input("Введите натуральное число b1: "))
                    b2 = int(input("Введите натуральное число b2: "))
                    m1 = int(input("Введите натуральное число m1: "))
                    m2 = int(input("Введите натуральное число m2: "))

                    # Решение системы линейных сравнений
                    try:
                        result = chinese_remainder_theorem(b1, b2, m1, m2)
                        print("Решение системы x ≡ {} (mod {})".format(result, m1 * m2))
                    except ValueError as e:
                        print(e)
                    except Exception as e:
                        print("Произошла ошибка:", e)
                case ['4']:
                    def generate_prime(bits):
                        while True:
                            num = random.getrandbits(bits)
                            if isprime(num):
                                return num

                    def generate_keypair(bits):
                        p = generate_prime(bits)
                        q = generate_prime(bits)
                        n = p * q
                        phi_n = (p - 1) * (q - 1)
                        e = 65537
                        d = mod_inverse(e, phi_n)

                        return n, e, d

                    bits = int(input("Введите количество битов для простых чисел: "))
                    n, e, d = generate_keypair(bits)
                    print("Открытый ключ (n, e): ({}, {})".format(n, e))
                    print("Закрытый ключ (n, d): ({}, {})".format(n, d))

                    def encrypt_decrypt_number(number, key, modulus):
                        return pow(number, key, modulus)

                    number = int(input("Введите натуральное число: "))
                    n = int(input("Введите n: "))
                    key = int(input("Введите e (для шифрования) или d (для дешифрования): "))
                    operation = input("Выберите операцию (encrypt/decrypt): ")

                    if operation == "encrypt":
                        result = encrypt_decrypt_number(number, key, n)
                        print("Зашифрованное число:", result)
                    elif operation == "decrypt":
                        result = encrypt_decrypt_number(number, key, n)
                        print("Дешифрованное число:", result)
                    else:
                        print("Неверная операция. Выберите 'encrypt' или 'decrypt'.")




