import unittest
import src.genetic as gen

class TestStringMethods(unittest.TestCase):

    def test_getAverageValue(self):
        lst = [1,2,3]
        self.assertEqual(gen.getAverageValue(lst), 2)

    def test_getAverageValue_emptyList(self):
        lst = []
        self.assertEqual(gen.getAverageValue(lst), 0)

if __name__ == '__main__':
    unittest.main()
