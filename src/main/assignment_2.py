import numpy as np
import unittest
import math


def neville(datax, datay, x):
    """
    Finds an interpolated value using Neville's algorithm.

    Input
      datax: input x's in a list of size n
      datay: input y's in a list of size n
      x: the x value used for interpolation

    Output
      p[0]: the polynomial of degree n
    """
    n = len(datax)
    p = n*[0]
    for k in range(n):
        for i in range(n-k):
            if k == 0:
                p[i] = datay[i]
            else:
                p[i] = ((x-datax[i+k])*p[i]+ \
                        (datax[i]-x)*p[i+1])/ \
                        (datax[i]-datax[i+k])
    return p[0]



def nevilles_method(x_points, y_points, x):
    n = len(x_points)
    Q = np.zeros((n, n))
    Q[:, 0] = y_points

    for j in range(1, n):
        for i in range(n - j):
            Q[i][j] = ((x - x_points[i + j]) * Q[i][j - 1] + (x_points[i] - x) * Q[i + 1][j - 1]) / (x_points[i] - x_points[i + j])

    return Q[0][-1]

# Task 2 & 3: Newton's Forward Method

def newtons_forward_method(x_points, y_points):
    n = len(x_points)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y_points

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i + 1][j - 1] - diff_table[i][j - 1]

    return diff_table



def interpolate_newton(x_points, y_points, x):
    diff_table = newtons_forward_method(x_points, y_points)
    n = len(x_points)
    h = x_points[1] - x_points[0]
    u = (x - x_points[0]) / h
    result = diff_table[0, 0]
    u_product = 1

    for i in range(1, n):
        u_product *= (u - (i - 1))
        result += (u_product * diff_table[0, i]) / math.factorial(i)  # Updated

    return result

# Task 4: Hermite Polynomial

def hermite_polynomial(x_points, y_points, y_derivatives):
    n = len(x_points)
    z = np.zeros(2 * n)
    Q = np.zeros((2 * n, 2 * n))

    for i in range(n):
        z[2 * i] = z[2 * i + 1] = x_points[i]
        Q[2 * i][0] = Q[2 * i + 1][0] = y_points[i]
        Q[2 * i + 1][1] = y_derivatives[i]
        if i != 0:
            Q[2 * i][1] = (Q[2 * i][0] - Q[2 * i - 1][0]) / (z[2 * i] - z[2 * i - 1])

    for j in range(2, 2 * n):
        for i in range(2 * n - j):
            Q[i][j] = (Q[i + 1][j - 1] - Q[i][j - 1]) / (z[i + j] - z[i])

    return Q

# Task 5: Cubic Spline Interpolation

def cubic_spline(x_points, y_points):
    n = len(x_points) - 1
    h = [x_points[i + 1] - x_points[i] for i in range(n)]
    alpha = [3 * (y_points[i + 1] - y_points[i]) / h[i] - 3 * (y_points[i] - y_points[i - 1]) / h[i - 1] for i in range(1, n)]

    l = [1] + [0] * n
    mu = [0] * n
    z = [0] * (n + 1)
    b = [0] * n
    c = [0] * (n + 1)
    d = [0] * n

    for i in range(1, n):
        l[i] = 2 * (x_points[i + 1] - x_points[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i - 1] - h[i - 1] * z[i - 1]) / l[i]

    l[n] = 1

    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y_points[j + 1] - y_points[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    return b, c, d


class TestAssignment2(unittest.TestCase):
    def test_nevilles_method(self):
        x = [3.6, 3.8, 3.9]
        y = [1.675, 1.436, 1.318]
        result = nevilles_method(x, y, 3.7)
        self.assertAlmostEqual(result, 1.555, places=3)

    def test_newtons_forward_method(self):
        x = [7.2, 7.4, 7.5, 7.6]
        y = [23.5492, 25.3913, 26.8224, 27.4589]
        diff_table = newtons_forward_method(x, y)
        self.assertAlmostEqual(diff_table[0][1], 9.2105, places=4)

    def test_interpolate_newton(self):
        x = [7.2, 7.4, 7.5, 7.6]
        y = [23.5492, 25.3913, 26.8224, 27.4589]
        result = interpolate_newton(x, y, 7.3)
        self.assertAlmostEqual(result, 24.5, places=1)

    def test_hermite_polynomial(self):
        x = [3.6, 3.8, 3.9]
        y = [1.675, 1.436, 1.318]
        dy = [-1.195, -1.188, -1.182]
        hermite_matrix = hermite_polynomial(x, y, dy)
        self.assertAlmostEqual(hermite_matrix[0][0], 1.675, places=3)

    def test_cubic_spline(self):
        x = [2, 5, 8, 10]
        y = [3, 5, 7, 9]
        b, c, d = cubic_spline(x, y)
        self.assertEqual(len(b), 3)
        self.assertEqual(len(c), 4)
        self.assertEqual(len(d), 3)

if __name__ == "__main__":
    unittest.main()