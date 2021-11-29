import unittest

import src.population_manipulation as pop

class MyTestCase(unittest.TestCase):
    def test_initialize_population_init_method(self):
        with self.assertRaises(Exception):
            pop.initialize_population(1, {'notImplemented': 1})


    def test_initialize_population_ratios(self):
        _, ratios = pop.initialize_population(90, {'kruscal': 1, 'kmedoid': 1, 'kmeans': 1})
        expected_ratios = {'kruscal': 30, 'kmedoid': 30, 'kmeans': 30}
        self.assertEqual(ratios, expected_ratios)