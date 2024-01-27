import numpy as np
import math


class Benchmark:
    NAME = "Benchmark"
    LOWER_BOUND = -100
    UPPER_BOUND = 100

    def __init__(self, index=0, max_runs=0):
        self.index = index
        self.max_runs = max_runs
        self.current_run = 0

    def get_index(self):
        return self.index

    def restart(self):
        self.current_run = 0

    def get_name(self):
        return self.NAME
        # return self.NAME if self.NAME is not None else "Undefined benchmark name"

    def get_lower_bound(self):
        return self.LOWER_BOUND
        # return self.LOWER_BOUND if self.LOWER_BOUND is not None else Settings.LOWER_BOUND

    def get_upper_bound(self):
        return self.UPPER_BOUND
        # return self.UPPER_BOUND if self.UPPER_BOUND is not None else Settings.UPPER_BOUND

    def evaluate(self, element):
        if self.max_runs == 0:
            return self._evaluate(element)

        self.current_run += 1
        if self.current_run > self.max_runs:
            raise StopIteration
        return self._evaluate(element)

    def _evaluate(self, element):
        raise NotImplementedError


class Rosenbrock(Benchmark):
    # https://www.sfu.ca/~ssurjano/rosen.html
    # global minimum at [0, ..., 0]
    NAME = "Rosenbrock"
    # LOWER_BOUND = -5
    # UPPER_BOUND = 10
    LOWER_BOUND = -2.48
    UPPER_BOUND = 2.48

    def _evaluate(self, x):
        result = 0
        for i in range(len(x) - 1):
            result += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2

        return result


class Schwefel(Benchmark):
    # global minimum at ~ [420,420] - not suitable for range (-100,100)
    NAME = "Schwefel"
    LOWER_BOUND = -500
    UPPER_BOUND = 500

    def _evaluate(self, x):
        n = len(x)
        sum_term = 0
        for i in range(n):
            sum_term += x[i] * math.sin(math.sqrt(abs(x[i])))

        return 418.9829 * n - sum_term


class Schwefel20(Benchmark):
    # https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12
    NAME = "Schwefel2"

    def _evaluate(self, x):
        return np.sum(np.abs(x))


class Schwefel23(Benchmark):
    # https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12
    NAME = "Schwefel23"
    LOWER_BOUND = -10
    UPPER_BOUND = 10

    def _evaluate(self, x):
        return np.sum(x ** 10)


class Matyas(Benchmark):
    NAME = "Matyas"

    def _evaluate(self, x):
        n = len(x)
        result = 0
        for i in range(n - 1):
            result += 0.26 * (x[i] ** 2 + x[i + 1] ** 2) - 0.48 * x[i] * x[i + 1]
        return result


class DeJongSphere(Benchmark):
    # global minimum at [0, ..., 0]
    NAME = "DeJong Sphere"

    def _evaluate(self, x):
        result = 0
        for i in range(len(x)):
            result += x[i] ** 2
        return result


class Styblinski(Benchmark):
    # https://www.sfu.ca/~ssurjano/stybtang.html
    # global minimum at [-2.903534, ..., -2.903534]
    NAME = "Styblinski"
    LOWER_BOUND = -5
    UPPER_BOUND = 5

    def _evaluate(self, x):
        result = 0
        for i in range(len(x)):
            result += (x[i] ** 4 - 16 * x[i] ** 2 + 5 * x[i])
        return 0.5 * result


class Rastrigin(Benchmark):
    # https://www.sfu.ca/~ssurjano/rastr.html
    # global minimum at [0, ..., 0]
    NAME = 'Rastrigin'
    LOWER_BOUND = -5.12
    UPPER_BOUND = 5.12

    def _evaluate(self, x, A=10):
        n = len(x)
        sum_term = np.sum(x ** 2 - A * np.cos(2 * np.pi * x))
        return A * n + sum_term


class Griewank(Benchmark):
    # https://www.sfu.ca/~ssurjano/griewank.html
    NAME = 'Griewank'

    def _evaluate(self, x):
        if x is not np.ndarray:
            x = np.array(x)
        n = len(x)
        sum_term = np.sum(x ** 2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
        return sum_term - prod_term + 1


class Ackley(Benchmark):
    # https://www.sfu.ca/~ssurjano/ackley.html
    NAME = 'Ackley'
    LOWER_BOUND = -10
    UPPER_BOUND = 10

    def _evaluate(self, x):
        if isinstance(x, list):
            x = np.array(x)
        n = len(x)
        term1 = np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
        term2 = np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
        return -20 * term1 - term2 + 20 + np.exp(1)


class Michalewicz(Benchmark):
    NAME = 'Michalewicz'
    LOWER_BOUND = 0
    UPPER_BOUND = np.pi

    def _evaluate(self, x, m=10):
        n = len(x)
        i = np.arange(1, n + 1)
        sin_terms = np.sin(x) * np.sin((i * x ** 2) / np.pi)
        return -np.sum(sin_terms)


class Michalewicz2(Benchmark):
    # https://www.sfu.ca/~ssurjano/michal.html
    NAME = 'Michalewicz2'
    LOWER_BOUND = 0
    UPPER_BOUND = np.pi

    def _evaluate(self, x, m=10):
        n = len(x)
        i = np.arange(1, n + 1)
        sin_terms = np.sin(x) * np.sin((i * x ** 2) / np.pi) ** (2 * m)
        return -np.sum(sin_terms)


class Alpine(Benchmark):
    # https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12
    NAME = 'Alpine'
    LOWER_BOUND = 0
    UPPER_BOUND = 10

    def _evaluate(self, x):
        # if type(x) is not np.ndarray:
        #     x = np.array(x)
        return np.sum(np.abs(x * np.sin(x) + 0.1 * x))


class Alpine2(Benchmark):
    # https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12
    NAME = 'Alpine2'
    LOWER_BOUND = 0
    UPPER_BOUND = 10

    def _evaluate(self, x):
        return -np.prod(np.sqrt(x) * np.sin(x))


class Qing(Benchmark):
    # https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12
    NAME = 'Qing'
    LOWER_BOUND = -2
    UPPER_BOUND = 2

    def _evaluate(self, x):
        n = len(x)
        total_sum = 0
        for i in range(n):
            total_sum += (x[i] ** 2 - i) ** 2

        return total_sum


class Quartic(Benchmark):
    # https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12
    NAME = 'Quartic'
    LOWER_BOUND = -1.28
    UPPER_BOUND = 1.28

    def _evaluate(self, x):
        n = len(x)
        total_sum = 0
        for i in range(n):
            total_sum += i * x[i] ** 4 + np.random.rand()

        return total_sum


class RotatedHyperEllipsoid(Benchmark):
    # link: https://www.sfu.ca/~ssurjano/rothyp.html
    NAME = 'RotatedHyperEllipsoid'

    def _evaluate(self, x):
        n = len(x)
        total_sum = 0
        for i in range(n):
            total_sum += np.sum(x[:i + 1] ** 2)
        return total_sum


class Salomon(Benchmark):
    # https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12
    NAME = 'Salomon'
    LOWER_BOUND = -np.pi
    UPPER_BOUND = np.pi

    def _evaluate(self, x):
        n = len(x)
        square_root_term = np.sqrt(np.sum(x ** 2))
        cos_term = np.cos(2 * np.pi * square_root_term)
        return 1 - cos_term + 0.1 * square_root_term


class Zakharov(Benchmark):
    # https://www.sfu.ca/~ssurjano/zakharov.html
    # https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12
    NAME = 'Zakharov'
    LOWER_BOUND = -5
    UPPER_BOUND = 10

    def _evaluate(self, x):
        n = len(x)
        i = np.arange(1, n + 1)
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(0.5 * i * x)
        return sum1 + sum2 ** 2 + sum2 ** 4


class DifferentPowersSum(Benchmark):
    # https://www.sfu.ca/~ssurjano/sumpow.html
    NAME = 'DifferentPowersSum'
    LOWER_BOUND = -1
    UPPER_BOUND = 1

    def _evaluate(self, x):
        n = len(x)
        i = np.arange(1, n + 1)
        absolute_powers = np.abs(x) ** (i + 1)
        return np.sum(absolute_powers)


class SumSquares(Benchmark):
    # https://www.sfu.ca/~ssurjano/sumsqu.html
    # https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12
    NAME = 'SumSquares'
    LOWER_BOUND = -10
    UPPER_BOUND = 10

    def _evaluate(self, x):
        n = len(x)
        i = np.arange(1, n + 1)
        return np.sum(i * x ** 2)


class Trid(Benchmark):
    # https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12
    NAME = 'Trid'

    def _evaluate(self, x):
        n = len(x)
        total_sum = 0
        for i in range(n):
            total_sum += np.sum(x[:i + 1] ** 2)
        return total_sum


class XinSheYang(Benchmark):
    # https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12
    NAME = 'Xin-She Yang'
    LOWER_BOUND = -5
    UPPER_BOUND = 5

    def _evaluate(self, x):
        n = len(x)
        i = np.arange(1, n + 1)

        return np.sum(np.random.rand() * np.abs(x) ** i)


class DixonPrice(Benchmark):
    # https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12
    NAME = 'DixonPrice'
    LOWER_BOUND = -10
    UPPER_BOUND = 10

    def _evaluate(self, x):
        term1 = (x[0] - 1) ** 2
        term2 = np.sum([i * (2 * x[i] ** 2 - x[i - 1]) ** 2 for i in range(1, len(x))])
        return term1 + term2


class Exponential(Benchmark):
    # https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12
    NAME = 'Exponential'
    LOWER_BOUND = -1
    UPPER_BOUND = 1

    def _evaluate(self, x):
        sum_of_squares = np.sum(np.square(x))
        return -np.exp(-0.5 * sum_of_squares)
