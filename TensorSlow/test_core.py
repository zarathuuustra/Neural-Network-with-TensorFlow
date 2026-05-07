import unittest
from core_simple import *


#Best Practices:
# 1) Test should be isolated; they shouldn't depend on each other
# 2) Test-driven development: Write the test before you write the code



class SquareTest(unittest.TestCase):
    """
    Convenient way to test in Python with unittest.
    """
    @classmethod
    def setUpClass(cls):
        """
        Wird einmal vor allem anderen aufgerufen
        :return:
        """
        print("Setting up...")

    @classmethod
    def tearDownClass(cls):
        """
        Wird einmal nach allem anderen aufgerufen
        :return:
        """
        print("Finished...")

    def setUp(self):
        """
        This method is called before every test.
        :return:
        """
        pass

    def tearDown(self):
        """
        This method is called after every test.
        :return:
        """
        pass

    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected) # assert to verify the output

    def test_backward(self):  # added
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_variable(self):
        """
        To see if the variable class functions properly
        :return:
        """
        self.assertRaises(TypeError, Variable, 2)


if __name__ == '__main__':
    unittest.main()
