import unittest
from GroupMaker import GroupMaker
import os
from DataLoader import AffinityMatrixGenerator

from tests.generate_data import generate_test_csv

class TestGroupMaker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        generate_test_csv('test_reponses.csv', num_students=100, max_wishes=2)

        cls.generate_test_csv()

    @classmethod
    def generate_test_csv(cls):
        generator = AffinityMatrixGenerator('test_reponses.csv')
        generator.load_csv()
        generator.generate_affinity_matrix()
        generator.save_matrix_csv("./affinity_matrix.csv")
        cls.affinity_matrix_path = "./affinity_matrix.csv"

    def setUp(self):
        self.group_size = 4
        self.group_maker = GroupMaker(self.affinity_matrix_path, self.group_size)

    def test_initial_random_groups(self):
        groups = self.group_maker.initial_random_groups()
        self.assertEqual(len(groups), len(self.group_maker._GroupMaker__students) // self.group_size)
        for group in groups:
            self.assertEqual(len(group), self.group_size)

    def test_hierarchical_clustering(self):
        groups = self.group_maker.hierarchical_clustering()
        self.assertTrue(all(len(group) == self.group_size for group in groups[:-1]))
        self.assertTrue(len(groups[-1]) <= self.group_size)

    def test_compute_total_score(self):
        self.group_maker.hierarchical_clustering()
        total_score = self.group_maker.compute_total_score()
        self.assertGreaterEqual(total_score, 0)
        self.assertLessEqual(total_score, 1)

    def test_save_groups(self):
        output_file = "./output_groups.csv"
        self.group_maker.hierarchical_clustering()
        self.group_maker.save_groups(output_file)
        self.assertTrue(os.path.exists(output_file))

if __name__ == "__main__":
    unittest.main()