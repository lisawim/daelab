import operator

from daelab.base.datatypes.arithmetic import ArithmeticMethods


class DummyMethod(ArithmeticMethods):
    def _binary(self, other, op):
        return op, other


def test_add_calls_binary_with_add():
    assert DummyMethod() + 2 == (operator.add, 2)


def test_radd_calls_binary_with_add():
    assert 3 + DummyMethod() == (operator.add, 3)


def test_sub_calls_binary_with_sub():
    assert DummyMethod() - 5 == (operator.sub, 5)


def test_mul_calls_binary_with_mul():
    assert DummyMethod() * 2.5 == (operator.mul, 2.5)


def test_rmul_calls_binary_with_mul():
    assert 2.5 * DummyMethod() == (operator.mul, 2.5)


def test_truediv_calls_binary_with_truediv():
    assert DummyMethod() / 4 == (operator.truediv, 4)


def test_neg_multiplies_with_minus_one():
    assert -DummyMethod() == (operator.mul, -1)
