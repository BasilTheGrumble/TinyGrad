import unittest
from TinyGrad import Value


class TestValue(unittest.TestCase):
    def test_add(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        c.backward()
        self.assertAlmostEqual(c.data, 5.0, places=4)
        self.assertAlmostEqual(a.grad, 1.0, places=4)
        self.assertAlmostEqual(b.grad, 1.0, places=4)

    def test_mul(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()
        self.assertAlmostEqual(c.data, 6.0, places=4)
        self.assertAlmostEqual(a.grad, 3.0, places=4)
        self.assertAlmostEqual(b.grad, 2.0, places=4)

    def test_div(self):
        a = Value(6.0)
        b = Value(2.0)
        c = a / b
        c.backward()
        self.assertAlmostEqual(c.data, 3.0, places=4)
        self.assertAlmostEqual(a.grad, 0.5, places=4)
        self.assertAlmostEqual(b.grad, -1.5, places=4)

    def test_relu(self):
        a = Value(-2.0)
        b = a.relu()
        b.backward()
        self.assertAlmostEqual(b.data, 0.0, places=4)
        self.assertAlmostEqual(a.grad, 0.0, places=4)

        a = Value(2.0)
        b = a.relu()
        b.backward()
        self.assertAlmostEqual(b.data, 2.0, places=4)
        self.assertAlmostEqual(a.grad, 1.0, places=4)

    def test_sigmoid(self):
        a = Value(0.0)
        b = a.sigmoid()
        b.backward()
        self.assertAlmostEqual(b.data, 0.5, places=4)
        self.assertAlmostEqual(a.grad, 0.25, places=4)

    def test_pow(self):
        a = Value(2.0)
        b = a ** 3
        b.backward()
        self.assertAlmostEqual(b.data, 8.0, places=4)
        self.assertAlmostEqual(a.grad, 12.0, places=4)  # 3*(2)^2 = 12

    def test_chain_rule(self):
        a = Value(2.0)
        b = a ** 2 + 1
        c = b.relu()
        d = c * 3
        d.backward()
        self.assertAlmostEqual(d.data, 15.0, places=4)  # (2²+1)*3=15
        self.assertAlmostEqual(a.grad, 12.0, places=4)   # 3*(2*2) = 12


if __name__ == '__main__':
    unittest.main()