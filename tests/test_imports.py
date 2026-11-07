import daelab
import daelab.problems
import daelab.methods


def test_package_import():
    assert daelab is not None


def test_problems_import():
    assert daelab.problems is not None


def test_methods_import():
    assert daelab.methods is not None
