import unittest
from assignment_2 import nevilles_method, newtons_forward_method, interpolate_newton, hermite_polynomial, cubic_spline

class TestAssignment2(unittest.TestCase):
    def test_nevilles_method(self):
        x = [3.6, 3.8, 3.9]
        y = [1.675, 1.436, 1.318]
        result = nevilles_method(x, y, 3.7)
        self.assertAlmostEqual(result, 1.553, places=3)

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